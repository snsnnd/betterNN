"""第三轮：I/R/H完整消融、W训练方式、提示可靠性。"""
import argparse,copy,itertools,json,time,csv
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from v2_reference import FlowNet as OriginalFlow,GRU,dataset,CONDITIONS,clock

# 前三位依次是Input/Route/Hold门；所有组均训练读出。
VARIANTS=[f'g{i}{r}{h}' for i,r,h in itertools.product([0,1],repeat=3)]+['g111_slow','g111_joint','gru']
# 提示模糊/移除还需从头训练对照，而非仅测试时破坏输入。
VARIANTS+=['g101_noisy','g101_none','gru_noisy','gru_none']

def cue_mode(v):return 'noisy' if v.endswith('_noisy') else ('none' if v.endswith('_none') else 'clean')
def corrupt(m,mode,seed):
    if mode=='none':return torch.zeros_like(m)
    if mode=='noisy':
        gen=torch.Generator().manual_seed(seed);flip=torch.rand(m.shape,generator=gen)<.15
        return torch.where(flip,1-m,m)
    return m

def inputs(data,mode,seed):
    x,q,m,y=data;return x,q,corrupt(m,mode,seed),y

class Flow(OriginalFlow):
    def __init__(self,v):
        # 保持B/C/W/控制器/读出初始化与第二轮一致。
        super().__init__('marker_memory');self.code=v
        self.I,self.R,self.H=[bool(int(c)) for c in v[1:4]]
        self.W.requires_grad_(v in ['g111_slow','g111_joint'])
        for mod,flag in [(self.upper,self.I),(self.lower,self.R),(self.update,self.H)]:
            for p in mod.parameters():p.requires_grad_(flag)
    def forward(self,x,q,m,trace=False):
        h=x.new_zeros(len(x),self.width);ps=[];aa=[]
        for t in range(x.shape[1]):
            meta=torch.cat([q,clock(x,t),m[:,t:t+1]],1)
            p=torch.sigmoid(self.upper(meta)) if self.I else torch.ones_like(q)
            loc=h.reshape(-1,4,self.width//4).mean(-1)
            g=torch.sigmoid(self.lower(torch.cat([q,loc],1))) if self.R else torch.ones_like(q)
            a=.2*torch.sigmoid(self.update(meta)) if self.H else x.new_full((len(x),1),.2)
            cand=torch.tanh((h*g.repeat_interleave(self.width//4,1))@(self.W*self.mask)+(x[:,t]*p)@self.B+meta@self.C)
            h=(1-a)*h+a*cand
            if trace:ps.append(p.detach());aa.append(a.detach())
        y=self.head(h).squeeze(-1)
        return (y,torch.stack(ps,1),torch.stack(aa,1)) if trace else y

def build(v):return GRU(8) if v.startswith('gru') else Flow(v)
@torch.no_grad()
def predict(model,data):
    x,q,m,y=data
    return torch.cat([model(x[i:i+512],q[i:i+512],m[i:i+512]) for i in range(0,len(x),512)])
def accuracy(model,data):return ((predict(model,data)>0)==data[3].bool()).float().mean().item()

def assert_frozen(model,initial):
    if isinstance(model,Flow):
        for k in ['B','C','mask']+([] if model.W.requires_grad else ['W']):assert torch.equal(model.state_dict()[k],initial[k]),k
        for name,enabled in [('upper',model.I),('lower',model.R),('update',model.H)]:
            if not enabled:
                for k,t in model.state_dict().items():
                    if k.startswith(name+'.'):assert torch.equal(t,initial[k]),k

@torch.no_grad()
def channel_diagnostics(model,data,truth_marker):
    x,q,m,y=data;logits,p,a=model(x,q,m,trace=True);res=[]
    for channel in range(4):
        ids=q[:,channel].bool();valid=truth_marker[ids].bool()
        res.append({'channel':channel,'accuracy':((logits[ids]>0)==y[ids].bool()).float().mean().item(),'valid_injection':p[ids,:,channel][valid].mean().item(),'invalid_injection':p[ids,:,channel][~valid].mean().item(),'valid_update':a[ids,:,0][valid].mean().item(),'invalid_update':a[ids,:,0][~valid].mean().item()})
    return res

@torch.no_grad()
def markerless_reference(data):
    # 已知两个位置均匀选自前6步。无标记时枚举15对，以多数标签作Bayes预测。
    x,q,_,y=data;v=x.gather(2,q.argmax(1)[:,None,None].expand(-1,x.shape[1],1)).squeeze(-1)
    labels=torch.stack([(v[:,i]+v[:,j])>0 for i,j in itertools.combinations(range(6),2)],1)
    prob=labels.float().mean(1);expected=torch.maximum(prob,1-prob).mean().item()
    acc=((prob>.5)==y.bool()).float().mean().item()
    return {'conditional_optimal_expected_accuracy':expected,'majority_rule_empirical_accuracy':acc}

def get_tests(seed,mode):
    raw={k:dataset(4096,seed*1000+10+i,**cfg) for i,(k,cfg) in enumerate(CONDITIONS.items())}
    result={k:inputs(d,mode,seed*1000+100+i) for i,(k,d) in enumerate(raw.items())}
    result['clean_id']=raw['id12']
    result['noisy_id']=inputs(raw['id12'],'noisy',seed*1000+100)
    result['none_id']=inputs(raw['id12'],'none',0)
    return raw,result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=100);ap.add_argument('--seeds',type=int,nargs='+',default=[11,22,33,44,55]);ap.add_argument('--variants',nargs='+',choices=VARIANTS,default=VARIANTS);ap.add_argument('--out',default='results');args=ap.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    config=vars(args)|{'train_n':3072,'val_n':768,'test_n':4096,'batch':256,'lr':.003,'slow_lr':.00015,'flip_probability':.15,'conditions':CONDITIONS,'torch':torch.__version__,'numpy':np.__version__,'threads':1}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    rows=[];histories={};diagnostics={};references={}
    for seed in args.seeds:
        rawtrain=dataset(3072,seed*1000);rawval=dataset(768,seed*1000+1)
        references[str(seed)]=markerless_reference(dataset(4096,seed*1000+10))
        for variant in args.variants:
            mode=cue_mode(variant);train=inputs(rawtrain,mode,seed*1000+200);val=inputs(rawval,mode,seed*1000+201);rawtests,tests=get_tests(seed,mode)
            torch.manual_seed(seed);model=build(variant);initial=copy.deepcopy(model.state_dict())
            groups=[{'params':[p for n,p in model.named_parameters() if p.requires_grad and n!='W'],'lr':.003}]
            if isinstance(model,Flow) and model.W.requires_grad:groups.append({'params':[model.W],'lr':.00015 if variant.endswith('_slow') else .003})
            opt=torch.optim.Adam(groups);gen=torch.Generator().manual_seed(seed+5000);best_acc=-1;best=None;best_epoch=0;history=[];gate_history=[]
            start=time.perf_counter()
            for epoch in range(args.epochs):
                model.train();order=torch.randperm(len(train[0]),generator=gen);loss_sum=0
                for ids in order.split(256):
                    opt.zero_grad(set_to_none=True);logit=model(train[0][ids],train[1][ids],train[2][ids]);loss=F.binary_cross_entropy_with_logits(logit,train[3][ids]);loss.backward()
                    nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);opt.step();loss_sum+=loss.item()*len(ids)
                model.eval();va=accuracy(model,val);history.append({'epoch':epoch+1,'loss':loss_sum/len(order),'val_accuracy':va})
                if va>best_acc:best_acc=va;best=copy.deepcopy(model.state_dict());best_epoch=epoch+1
                if isinstance(model,Flow) and (epoch==0 or (epoch+1)%10==0):
                    ds=tuple(t[:256] for t in train);gate_history.append({'epoch':epoch+1,'channels':channel_diagnostics(model,ds,rawtrain[2][:256])})
            elapsed=time.perf_counter()-start;model.load_state_dict(best);model.eval();assert_frozen(model,initial)
            row={'seed':seed,'variant':variant,'train_cue':mode,'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),'best_epoch':best_epoch,'val_accuracy':best_acc,'train_seconds':elapsed}
            row.update({k:accuracy(model,d) for k,d in tests.items()})
            if isinstance(model,Flow):
                diag=channel_diagnostics(model,tuple(t[:1024] for t in tests['id12']),rawtests['id12'][2][:1024]);diagnostics[f'{variant}_{seed}']=diag
                row['min_channel_accuracy']=min(d['accuracy'] for d in diag)
                row['closed_channels']=sum(d['valid_injection']<.01 for d in diag) if model.I else 0
                row['backbone_max_change']=(model.W-initial['W']).abs().max().item()
            rows.append(row);histories[f'{variant}_{seed}']={'epochs':history,'gate_history':gate_history}
            torch.save({'variant':variant,'seed':seed,'state_dict':best},out/f'{variant}_seed{seed}.pt')
            for fname,obj in [('metrics',rows),('history',histories),('channel_diagnostics',diagnostics),('markerless_reference',references)]:
                (out/f'{fname}.json').write_text(json.dumps(obj,indent=2 if fname!='history' else None))
            print(f"seed={seed} {variant}: ID={row['id12']:.4f}, shift={row['shift12']:.4f}, delay={row['delay24']:.4f}, seconds={elapsed:.1f}",flush=True)
    summary={}
    for v in args.variants:
        rr=[r for r in rows if r['variant']==v];summary[v]={}
        for k,val in rr[0].items():
            if isinstance(val,(int,float)) and k!='seed':
                a=[r[k] for r in rr];summary[v][k]={'mean':float(np.mean(a)),'std':float(np.std(a,ddof=1)) if len(a)>1 else 0.}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    print('COMPLETE',flush=True)
if __name__=='__main__':main()

"""第四轮三个独立研究：mask控制、可学习显著性、延迟规则。"""
import argparse,copy,json,time,itertools
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.stats import rankdata
from legacy_experiment import Flow as LegacyFlow,dataset as old_dataset

class TopologyFlow(LegacyFlow):
    def __init__(self,mode,weight_seed,mask_seed):
        torch.manual_seed(weight_seed);super().__init__('g101' if mode=='no_route' else 'g111');self.mode=mode
        gen=torch.Generator().manual_seed(mask_seed);mask=(torch.rand(64,64,generator=gen)<.2).float()
        # 同一weight_seed下所有mask共享完整候选权重；只改变mask及范数归一化。
        gen=torch.Generator().manual_seed(weight_seed+10000);raw=torch.randn(64,64,generator=gen);w=raw*mask
        with torch.no_grad():self.mask.copy_(mask);self.W.copy_(w*.9/torch.linalg.matrix_norm(w,2))
    def forward(self,x,q,m):
        h=x.new_zeros(len(x),64)
        for t in range(x.shape[1]):
            meta=torch.cat([q,x.new_full((len(x),1),t/(x.shape[1]-1)),m[:,t:t+1]],1);p=self.upper(meta).sigmoid();a=.2*self.update(meta).sigmoid();loc=h.reshape(-1,4,16).mean(-1)
            gateinput=torch.cat([q,loc],1)
            if self.mode=='static':gateinput=torch.zeros_like(gateinput)
            g=self.lower(gateinput).sigmoid() if self.R else torch.ones_like(q)
            h=(1-a)*h+a*torch.tanh((h*g.repeat_interleave(16,1))@(self.W*self.mask)+(x[:,t]*p)@self.B+meta@self.C)
        return self.head(h).squeeze(-1)


def salience_data(n,seed,steps=12,region='early'):
    # 数值及标签完全延续旧任务；用可观察的事件键替代valid bit。
    x,q,mark,y=old_dataset(n,seed,steps=steps,region=region)
    gen=torch.Generator().manual_seed(seed+800000);target=q.argmax(1)
    keys=(target[:,None]+torch.randint(1,4,(n,steps),generator=gen))%4
    keys=torch.where(mark.bool(),target[:,None],keys);keys=F.one_hot(keys,4).float()
    flip=torch.rand(mark.shape,generator=gen)<.15;noisy=torch.where(flip,1-mark,mark)
    return (x,q,keys,noisy),y,{'valid':mark}

class SalienceFlow(LegacyFlow):
    def __init__(self,mode):
        super().__init__('g111');self.mode=mode
        self.register_buffer('E',torch.randn(4,64)*.15)
        self.salience=nn.Sequential(nn.Linear(8,16),nn.Tanh(),nn.Linear(16,1))
        for p in self.salience.parameters():p.requires_grad_(mode=='learned')
    def forward(self,x,q,keys,noisy,trace=False):
        h=x.new_zeros(len(x),64);ss=[]
        for t in range(x.shape[1]):
            if self.mode=='perfect':s=(q*keys[:,t]).sum(1,keepdim=True)
            elif self.mode=='noisy':s=noisy[:,t:t+1]
            elif self.mode=='none':s=x.new_zeros(len(x),1)
            else:s=self.salience(torch.cat([q,keys[:,t]],1)).sigmoid()
            meta=torch.cat([q,x.new_full((len(x),1),t/(x.shape[1]-1)),s],1);p=self.upper(meta).sigmoid();a=.2*self.update(meta).sigmoid();g=self.lower(torch.cat([q,h.reshape(-1,4,16).mean(-1)],1)).sigmoid()
            h=(1-a)*h+a*torch.tanh((h*g.repeat_interleave(16,1))@(self.W*self.mask)+(x[:,t]*p)@self.B+meta@self.C+keys[:,t]@self.E);ss.append(s)
        logits=self.head(h).squeeze(-1)
        return (logits,torch.stack(ss,1).squeeze(-1)) if trace else logits

class SalienceGRU(nn.Module):
    def __init__(self):super().__init__();self.rnn=nn.GRU(13,10,batch_first=True);self.head=nn.Linear(10,1)
    def forward(self,x,q,keys,noisy):
        tm=torch.linspace(0,1,x.shape[1]).view(1,-1,1).expand(len(x),-1,-1);inp=torch.cat([x,q[:,None].expand(-1,x.shape[1],-1),keys,tm],-1)
        h,_=self.rnn(inp);return self.head(h[:,-1]).squeeze(-1)


def delay_data(n,seed,steps=12,rule_range=(6,10)):
    gen=torch.Generator().manual_seed(seed);ab=torch.randn(n,2,generator=gen);rule=torch.randint(2,(n,),generator=gen);when=torch.randint(*rule_range,(n,),generator=gen)
    x=torch.zeros(n,steps,4);x[:,0,0]=ab[:,0];x[:,1,1]=ab[:,1]
    # meta = [data,rule,idle,rule0,rule1,time]，规则出现前没有任何规则输入。
    meta=torch.zeros(n,steps,6);meta[:,:,2]=1;meta[:,:2,0]=1;meta[:,:2,2]=0;meta[:,:,5]=torch.linspace(0,1,steps)
    idx=torch.arange(n);meta[idx,when,2]=0;meta[idx,when,1]=1;meta[idx,when,3+rule]=1
    base=(ab>0).float();y=torch.where(rule[:,None].bool(),base.flip(1),base)
    return (x,meta),y,{'rule':rule,'when':when,'opposite':base[:,0]!=base[:,1]}

class DelayFlow(LegacyFlow):
    def __init__(self,mode):
        super().__init__('g101' if mode=='no_route' else 'g111');self.mode=mode
        self.lower=nn.Sequential(nn.Linear(10,16),nn.Tanh(),nn.Linear(16,4));self.head=nn.Linear(64,2)
        for p in self.lower.parameters():p.requires_grad_(self.R)
        if mode=='slow':self.W.requires_grad_(True)
    def forward(self,x,meta,trace=False):
        h=x.new_zeros(len(x),64);gg=[]
        for t in range(x.shape[1]):
            mt=meta[:,t];p=self.upper(mt).sigmoid();a=.2*self.update(mt).sigmoid();loc=h.reshape(-1,4,16).mean(-1)
            gateinput=torch.cat([mt,loc],1)
            if self.mode=='static':gateinput=torch.zeros_like(gateinput)
            g=self.lower(gateinput).sigmoid() if self.R else torch.ones_like(h[:,:4])
            h=(1-a)*h+a*torch.tanh((h*g.repeat_interleave(16,1))@(self.W*self.mask)+(x[:,t]*p)@self.B+mt@self.C);gg.append(g)
        logits=self.head(h)
        return (logits,torch.stack(gg,1)) if trace else logits

class BlockDelayFlow(DelayFlow):
    def __init__(self,mode):
        super().__init__('dynamic');self.mode=mode
        # 保留与原动态模型相同的B/C/W、Write/Hold及读出初始化。
        self.lower=nn.Sequential(nn.Linear(10,16),nn.Tanh(),nn.Linear(16,16))
    def forward(self,x,meta,trace=False):
        h=x.new_zeros(len(x),64);gg=[];w=(self.W*self.mask).reshape(4,16,4,16)
        for t in range(x.shape[1]):
            mt=meta[:,t];p=self.upper(mt).sigmoid();a=.2*self.update(mt).sigmoid();loc=h.reshape(-1,4,16).mean(-1);gateinput=torch.cat([mt,loc],1)
            if self.mode=='block_static':gateinput=torch.zeros_like(gateinput)
            g=self.lower(gateinput).sigmoid().reshape(-1,4,4)
            messages=torch.einsum('nai,aibj->nabj',h.reshape(-1,4,16),w)
            recurrent=(messages*g[:,:,:,None]).sum(1).reshape(-1,64)
            h=(1-a)*h+a*torch.tanh(recurrent+(x[:,t]*p)@self.B+mt@self.C);gg.append(g)
        logits=self.head(h)
        return (logits,torch.stack(gg,1)) if trace else logits

class DelayGRU(nn.Module):
    def __init__(self):super().__init__();self.rnn=nn.GRU(10,10,batch_first=True);self.head=nn.Linear(10,2)
    def forward(self,x,meta):h,_=self.rnn(torch.cat([x,meta],-1));return self.head(h[:,-1])


def jobs_for(study):
    if study=='delay_block':return [{'study':study,'mode':m,'seed':s} for s in [11,22,33] for m in ['block_dynamic','block_static']]
    if study=='topology':
        jobs=[{'study':study,'mode':mode,'seed':ws,'mask_seed':ms} for ws in [101,202] for ms in [11,22,33,44,55] for mode in ['no_route','dynamic']]
        jobs +=[{'study':study,'mode':'static','seed':101,'mask_seed':ms} for ms in [11,22,33,44,55]]
        return jobs
    modes=['perfect','noisy','learned','none','gru'] if study=='salience' else ['no_route','dynamic','static','slow','gru']
    return [{'study':study,'mode':mode,'seed':s} for s in [11,22,33] for mode in modes]

def job_id(job):return '_'.join(str(job[k]) for k in ['study','mode','seed']+(['mask_seed'] if 'mask_seed' in job else []))
def build(job):
    torch.manual_seed(job['seed']);st=job['study'];m=job['mode']
    if st=='topology':return TopologyFlow(m,job['seed'],job['mask_seed'])
    if st=='delay_block':return BlockDelayFlow(m)
    if st=='salience':return SalienceGRU() if m=='gru' else SalienceFlow(m)
    return DelayGRU() if m=='gru' else DelayFlow(m)

def datasets(job):
    st=job['study'];seed=42000 if st=='topology' else job['seed']*1000
    if st=='topology':
        conv=lambda n,s,**kw:(lambda d:(d[:3],d[3],{}))(old_dataset(n,s,**kw))
        tr=conv(3072,seed);va=conv(768,seed+1);te={'id':conv(4096,seed+10),'shift':conv(4096,seed+11,region='late'),'delay':conv(4096,seed+12,steps=24)}
    elif st=='salience':
        tr=salience_data(3072,seed);va=salience_data(768,seed+1);te={'id':salience_data(4096,seed+10),'shift':salience_data(4096,seed+11,region='late'),'delay':salience_data(4096,seed+12,steps=24)}
    else:
        tr=delay_data(3072,seed);va=delay_data(768,seed+1)
        if st=='delay_block':seed+=700000
        te={'id':delay_data(4096,seed+10),'early_rule':delay_data(4096,seed+11,rule_range=(2,6)),'late_rule':delay_data(4096,seed+12,steps=24,rule_range=(18,22)),'long_hold':delay_data(4096,seed+13,steps=24)}
    return tr,va,te

@torch.no_grad()
def logits(model,data):
    inp,y,_=data;return torch.cat([model(*(x[i:i+512] for x in inp)) for i in range(0,len(y),512)])
def score(model,data):
    p=logits(model,data)>0;y=data[1].bool()
    return (p==y).all(1).float().mean().item() if y.ndim==2 else (p==y).float().mean().item()

def auc(scores,labels):
    sc=scores.detach().numpy().ravel();lb=labels.numpy().ravel().astype(bool);n1=lb.sum();n0=len(lb)-n1
    return float((rankdata(sc)[lb].sum()-n1*(n1+1)/2)/(n1*n0))

@torch.no_grad()
def diagnostics(model,job,data):
    inp,y,info=data;r={}
    if job['study']=='salience' and isinstance(model,SalienceFlow):
        _,s=model(*(x[:512] for x in inp),trace=True);m=info['valid'][:512].bool()
        r={'salience_auc':auc(s,m),'valid_salience':s[m].mean().item(),'invalid_salience':s[~m].mean().item()}
    if job['study'] in ['delay','delay_block']:
        p=logits(model,data)>0;correct=(p==y.bool()).all(1);r={'bit_accuracy':(p==y.bool()).float().mean().item(),'opposite_exact_accuracy':correct[info['opposite']].float().mean().item()}
        # 保持A/B和规则出现时刻不变，只翻转规则；双方都对才记成功。
        x,meta=inp;flipped=meta.clone();flipped[:,:,3:5]=meta[:,:,3:5].flip(-1);p2=logits(model,((x,flipped),y.flip(1),info))>0
        r['counterfactual_both_correct']=float((correct&(p2==y.flip(1).bool()).all(1)).float().mean())
        if isinstance(model,DelayFlow):
            _,g0=model(x[:512],meta[:512],trace=True);_,g1=model(x[:512],flipped[:512],trace=True);r['rule_gate_distance']=float((g0-g1).abs().mean())
    return r

def verify_frozen(model,initial):
    for k in ['B','C','E','mask','W']:
        if k in initial and not (k=='W' and model.W.requires_grad):assert torch.equal(model.state_dict()[k],initial[k]),k

def run(args):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);root=Path('results')/args.study;root.mkdir(parents=True,exist_ok=True)
    jobs=jobs_for(args.study);config={'study':args.study,'jobs':jobs,'epochs':args.epochs,'train_n':3072,'val_n':768,'test_n':4096,'batch':256,'lr':.003,'slow_lr':.00015,'torch':torch.__version__};(root/'config.json').write_text(json.dumps(config,indent=2));rows=[];histories={}
    for job in jobs:
        model=build(job);initial=copy.deepcopy(model.state_dict());train,val,tests=datasets(job)
        groups=[{'params':[p for n,p in model.named_parameters() if p.requires_grad and n!='W'],'lr':.003}]
        if hasattr(model,'W') and model.W.requires_grad:groups.append({'params':[model.W],'lr':.00015})
        opt=torch.optim.Adam(groups);gen=torch.Generator().manual_seed(5000 if args.study=='topology' else job['seed']+5000);best=-1;state=None;hist=[];start=time.perf_counter()
        for ep in range(args.epochs):
            model.train();order=torch.randperm(len(train[1]),generator=gen);total=0
            for ids in order.split(256):
                opt.zero_grad(set_to_none=True);pred=model(*(x[ids] for x in train[0]));loss=F.binary_cross_entropy_with_logits(pred,train[1][ids]);loss.backward();nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);opt.step();total+=loss.item()*len(ids)
            model.eval();va=score(model,val);hist.append({'epoch':ep+1,'loss':total/len(order),'validation':va})
            if va>best:best=va;state=copy.deepcopy(model.state_dict());bestep=ep+1
        seconds=time.perf_counter()-start;model.load_state_dict(state);verify_frozen(model,initial);model.eval()
        r=dict(job,run_id=job_id(job),trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),seconds=seconds,best_epoch=bestep,val_accuracy=best)
        for name,data in tests.items():r[name]=score(model,data)
        r.update(diagnostics(model,job,tests['id']))
        rows.append(r);histories[r['run_id']]=hist;torch.save({'job':job,'state_dict':state},root/(r['run_id']+'.pt'))
        (root/'metrics.json').write_text(json.dumps(rows,indent=2));(root/'history.json').write_text(json.dumps(histories));print(json.dumps(r),flush=True)
    print(args.study,'COMPLETE',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--study',choices=['topology','salience','delay','delay_block'],required=True);ap.add_argument('--epochs',type=int,default=100);run(ap.parse_args())

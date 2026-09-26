"""第二轮：随机有效时刻、位置迁移、延迟记忆、干扰增强。CPU 可复现。"""
import argparse, copy, csv, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

VARIANTS=['readout_only','clock_flow','marker_upper','marker_flow','marker_memory','slow_backbone','controller_only','gru_small','gru64']
EXTRA_VARIANTS=['marker_slowleak']
CONDITIONS={
 'id12':dict(steps=12,region='early',noise=1.),
 'shift12':dict(steps=12,region='late',noise=1.),
 'random12':dict(steps=12,region='all',noise=1.),
 'delay24':dict(steps=24,region='early',noise=1.),
 'noise12':dict(steps=12,region='early',noise=3.),
}

def dataset(n, seed, steps=12, region='early', noise=1.):
    gen=torch.Generator().manual_seed(seed)
    x=torch.randn(n,steps,4,generator=gen)
    q=F.one_hot(torch.randint(4,(n,),generator=gen),4).float()
    # 每例恰好两个有效时刻；训练时在前六步随机选，位置与信号值独立。
    allowed=torch.arange(6) if region=='early' else (torch.arange(steps-6,steps) if region=='late' else torch.arange(steps))
    ids=allowed[torch.rand(n,len(allowed),generator=gen).argsort(1)[:,:2]]
    mark=torch.zeros(n,steps);mark.scatter_(1,ids,1.)
    x=x*torch.where(mark[:,:,None].bool(),1.,noise)
    target=(x*mark[:,:,None]*q[:,None,:]).sum((1,2))
    return x,q,mark,(target>0).float()

def clock(x,t):return x.new_full((x.shape[0],1),t/(x.shape[1]-1))

class FlowNet(nn.Module):
    def __init__(self,variant,width=64):
        super().__init__();self.variant=variant;self.width=width
        mask=(torch.rand(width,width)<.2).float();w=torch.randn(width,width)*mask
        self.register_buffer('mask',mask)
        self.W=nn.Parameter(w*(.9/torch.linalg.matrix_norm(w,2)),requires_grad=variant=='slow_backbone')
        self.register_buffer('B',torch.randn(4,width)/2)
        # 所有 Flow 对照都获得 q、有效标记和时间，避免凭额外观测取得优势。
        self.register_buffer('C',torch.randn(6,width)*.15)
        self.upper=nn.Sequential(nn.Linear(6,16),nn.Tanh(),nn.Linear(16,4))
        self.lower=nn.Sequential(nn.Linear(8,16),nn.Tanh(),nn.Linear(16,4))
        self.head=nn.Linear(width,1)
        with torch.no_grad():self.head.weight.mul_(5)
        # 独立小保持门；只看任务、时间、标记，不看信号或标签。
        self.update=nn.Linear(6,1)
        nn.init.zeros_(self.update.weight);nn.init.zeros_(self.update.bias)
        self.use_upper=variant!='readout_only'
        self.use_lower=variant not in ['readout_only','marker_upper']
        for p in self.upper.parameters():p.requires_grad_(self.use_upper)
        for p in self.lower.parameters():p.requires_grad_(self.use_lower)
        for p in self.head.parameters():p.requires_grad_(variant!='controller_only')
        for p in self.update.parameters():p.requires_grad_(variant=='marker_memory')
    def forward(self,x,q,mark,force=None,trace=False):
        h=x.new_zeros(x.shape[0],self.width);ps=[];gs=[];us=[]
        for t in range(x.shape[1]):
            m=mark[:,t:t+1];tm=clock(x,t);meta=torch.cat([q,tm,m],1)
            # clock_flow：只移除上层对标记的直接访问；底层仍获得真实标记。
            ctlmeta=torch.cat([q,tm,torch.zeros_like(m)],1) if self.variant=='clock_flow' else meta
            p=torch.sigmoid(self.upper(ctlmeta)) if self.use_upper else torch.ones_like(q)
            local=h.reshape(-1,4,self.width//4).mean(-1)
            g=torch.sigmoid(self.lower(torch.cat([q,local],1))) if self.use_lower else torch.ones_like(q)
            a=.2*torch.sigmoid(self.update(meta)) if self.variant=='marker_memory' else x.new_full((x.shape[0],1),.1 if self.variant=='marker_slowleak' else .2)
            if force=='upper_open':p=torch.ones_like(p)
            if force=='lower_open':g=torch.ones_like(g)
            if force=='update_fixed':a=torch.full_like(a,.2)
            candidate=torch.tanh((h*g.repeat_interleave(self.width//4,1))@(self.W*self.mask)+(x[:,t]*p)@self.B+meta@self.C)
            h=(1-a)*h+a*candidate
            if trace:ps.append(p.detach());gs.append(g.detach());us.append(a.detach())
        logits=self.head(h).squeeze(-1)
        if trace:return logits,torch.stack(ps,1),torch.stack(gs,1),torch.stack(us,1)
        return logits

class GRU(nn.Module):
    def __init__(self,width):
        super().__init__();self.rnn=nn.GRU(10,width,batch_first=True);self.head=nn.Linear(width,1)
    def forward(self,x,q,mark):
        tm=torch.linspace(0,1,x.shape[1],device=x.device).view(1,-1,1).expand(x.shape[0],-1,-1)
        inp=torch.cat([x,q[:,None,:].expand(-1,x.shape[1],-1),tm,mark[:,:,None]],-1)
        h,_=self.rnn(inp);return self.head(h[:,-1]).squeeze(-1)

def build(variant):return GRU(8 if variant=='gru_small' else 64) if variant.startswith('gru') else FlowNet(variant)

@torch.no_grad()
def evaluate(model,data,force=None,zero_marker=False):
    x,q,m,y=data;pred=[]
    for i in range(0,len(x),512):
        mm=torch.zeros_like(m[i:i+512]) if zero_marker else m[i:i+512]
        logits=model(x[i:i+512],q[i:i+512],mm,force=force) if isinstance(model,FlowNet) else model(x[i:i+512],q[i:i+512],mm)
        pred.append(logits>0)
    return (torch.cat(pred)==y.bool()).float().mean().item()

@torch.no_grad()
def diagnostics(model,data):
    x,q,m,y=data;x,q,m=x[:1024],q[:1024],m[:1024]
    _,p,g,a=model(x,q,m,trace=True)
    selected=p.gather(2,q.argmax(1)[:,None,None].expand(-1,x.shape[1],1)).squeeze(-1)
    is_target=q[:,None,:].bool().expand_as(p)
    other=p[~is_target].reshape(x.shape[0],x.shape[1],3).mean(-1)
    valid=m.bool()
    return {'injection_valid_target':selected[valid].mean().item(),'injection_invalid_target':selected[~valid].mean().item(),'injection_valid_other':other[valid].mean().item(),'update_valid':a.squeeze(-1)[valid].mean().item(),'update_invalid':a.squeeze(-1)[~valid].mean().item(),'mean_gate':g.mean().item()}

def assert_frozen(model,initial):
    if not isinstance(model,FlowNet):return
    for k in ['B','C','mask']+([] if model.variant=='slow_backbone' else ['W']):assert torch.equal(model.state_dict()[k],initial[k]),k
    if model.variant=='controller_only':
        for k in ['head.weight','head.bias']:assert torch.equal(model.state_dict()[k],initial[k]),k

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=100);ap.add_argument('--seeds',nargs='+',type=int,default=[11,22,33]);ap.add_argument('--out',default='results');ap.add_argument('--variants',nargs='+',choices=VARIANTS+EXTRA_VARIANTS,default=VARIANTS);args=ap.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    config=vars(args)|{'train_n':3072,'val_n':768,'test_n':4096,'batch':256,'lr':.003,'backbone_lr':.00015,'conditions':CONDITIONS,'torch':torch.__version__,'numpy':np.__version__}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    rows=[];histories={}
    for seed in args.seeds:
        train=dataset(3072,seed*1000);val=dataset(768,seed*1000+1)
        tests={k:dataset(4096,seed*1000+10+i,**cfg) for i,(k,cfg) in enumerate(CONDITIONS.items())}
        for variant in args.variants:
            torch.manual_seed(seed);model=build(variant);initial=copy.deepcopy(model.state_dict())
            groups=[{'params':[p for n,p in model.named_parameters() if p.requires_grad and n!='W'],'lr':.003}]
            if variant=='slow_backbone':groups.append({'params':[model.W],'lr':.00015})
            opt=torch.optim.Adam(groups);gen=torch.Generator().manual_seed(seed+5000)
            best_acc=-1;best=None;best_epoch=0;history=[];start=time.perf_counter()
            for epoch in range(args.epochs):
                model.train();order=torch.randperm(len(train[0]),generator=gen);loss_sum=0
                for ids in order.split(256):
                    opt.zero_grad(set_to_none=True);loss=F.binary_cross_entropy_with_logits(model(train[0][ids],train[1][ids],train[2][ids]),train[3][ids]);loss.backward()
                    nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);opt.step();loss_sum+=loss.item()*len(ids)
                model.eval();va=evaluate(model,val);history.append({'epoch':epoch+1,'train_loss':loss_sum/len(order),'val_accuracy':va})
                if va>best_acc:best_acc=va;best=copy.deepcopy(model.state_dict());best_epoch=epoch+1
            elapsed=time.perf_counter()-start;model.load_state_dict(best);model.eval();assert_frozen(model,initial)
            r={'seed':seed,'variant':variant,'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),'registered_parameters':sum(p.numel() for p in model.parameters()),'train_seconds':elapsed,'best_epoch':best_epoch,'val_accuracy':best_acc}
            r.update({k:evaluate(model,d) for k,d in tests.items()});r['zero_marker_id12']=evaluate(model,tests['id12'],zero_marker=True)
            if isinstance(model,FlowNet):
                r.update(diagnostics(model,tests['id12']));r['backbone_max_change']=(model.W-initial['W']).abs().max().item()
                if variant in ['marker_flow','marker_memory']:
                    for cond in ['id12','delay24']:
                        for force in ['upper_open','lower_open']+(['update_fixed'] if variant=='marker_memory' else []):r[f'{cond}_{force}']=evaluate(model,tests[cond],force=force)
            rows.append(r);histories[f'{variant}_{seed}']=history
            torch.save({'seed':seed,'variant':variant,'state_dict':best},out/f'{variant}_seed{seed}.pt')
            (out/'metrics.json').write_text(json.dumps(rows,indent=2));(out/'history.json').write_text(json.dumps(histories))
            print(json.dumps(r),flush=True)
    summary={}
    for v in args.variants:
        rr=[r for r in rows if r['variant']==v];summary[v]={}
        for key in rr[0]:
            if key not in ['seed','variant']:
                values=[r[key] for r in rr];summary[v][key]={'mean':float(np.mean(values)),'std':float(np.std(values,ddof=1)) if len(values)>1 else 0.}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with (out/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    print('COMPLETE',flush=True)
if __name__=='__main__':main()

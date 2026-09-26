"""Salience实验：相关性由可观察key与query匹配决定，无随机隐藏标记。"""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.stats import rankdata

VARIANTS=['perfect','noisy','learned','supervised','none','gru']
CONDITIONS={'id':(12,'early',1.),'shift':(12,'late',1.),'random':(12,'all',1.),'delay':(24,'early',1.),'noise':(12,'early',3.)}

def dataset(n,seed,steps=12,region='early',noise=1.):
    gen=torch.Generator().manual_seed(seed);value=torch.randn(n,steps,1,generator=gen);query=torch.randint(4,(n,),generator=gen)
    keys=(query[:,None]+torch.randint(1,4,(n,steps),generator=gen))%4
    allowed=torch.arange(6) if region=='early' else (torch.arange(steps-6,steps) if region=='late' else torch.arange(steps))
    selected=allowed[torch.rand(n,len(allowed),generator=gen).argsort(1)[:,:2]]
    keys.scatter_(1,selected,query[:,None].expand(-1,2))
    relevance=(keys==query[:,None]).float().unsqueeze(-1)
    value=value*torch.where(relevance.bool(),1.,noise)
    y=((value*relevance).sum((1,2))>0).float()
    noisy=torch.where(torch.rand(relevance.shape,generator=gen)<.15,1-relevance,relevance)
    return value,F.one_hot(keys,4).float(),F.one_hot(query,4).float(),relevance,noisy,y

class SalienceNet(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind;h=64
        mask=(torch.rand(h,h)<.2).float();w=torch.randn(h,h)*mask
        self.register_buffer('mask',mask);self.W=nn.Parameter(.9*w/torch.linalg.matrix_norm(w,2))
        self.register_buffer('B',torch.randn(1,h)*.5);self.register_buffer('C',torch.randn(10,h)*.15)
        self.write=nn.Sequential(nn.Linear(6,16),nn.Tanh(),nn.Linear(16,1))
        self.route=nn.Sequential(nn.Linear(8,16),nn.Tanh(),nn.Linear(16,4))
        self.hold=nn.Linear(6,1);nn.init.zeros_(self.hold.weight);nn.init.zeros_(self.hold.bias)
        self.readout=nn.Linear(h,1)
        with torch.no_grad():self.readout.weight.mul_(5)
        self.salience=nn.Sequential(nn.Linear(8,16),nn.Tanh(),nn.Linear(16,1))
        for p in self.salience.parameters():p.requires_grad_(kind in ['learned','supervised'])
    def learned_scores(self,key,q):
        return self.salience(torch.cat([key,q[:,None,:].expand(-1,key.shape[1],-1)],-1)).sigmoid()
    def forward(self,value,key,q,hint,return_scores=False,force_score=None):
        if self.kind in ['learned','supervised']:s=self.learned_scores(key,q)
        elif self.kind in ['perfect','noisy']:s=hint
        else:s=torch.zeros_like(hint)
        if force_score is not None:s=torch.full_like(s,force_score)
        h=value.new_zeros(len(value),64)
        for t in range(value.shape[1]):
            tm=value.new_full((len(value),1),t/(value.shape[1]-1));ctl=torch.cat([q,tm,s[:,t]],1)
            p=self.write(ctl).sigmoid();g=self.route(torch.cat([q,h.reshape(-1,4,16).mean(-1)],1)).sigmoid();a=.2*self.hold(ctl).sigmoid()
            # 所有组底层都看原始key/query；none仅没有显式相关性标量。
            meta=torch.cat([q,key[:,t],tm,s[:,t]],1)
            candidate=torch.tanh((h*g.repeat_interleave(16,1))@(self.W*self.mask)+(value[:,t]*p)@self.B+meta@self.C)
            h=(1-a)*h+a*candidate
        logits=self.readout(h).squeeze(-1)
        return (logits,s) if return_scores else logits

class RawGRU(nn.Module):
    def __init__(self):
        super().__init__();self.gru=nn.GRU(10,32,batch_first=True);self.head=nn.Linear(32,1)
    def forward(self,value,key,q,hint):
        clock=torch.linspace(0,1,value.shape[1]).view(1,-1,1).expand(len(value),-1,-1)
        x=torch.cat([value,key,q[:,None].expand(-1,value.shape[1],-1),clock],-1);h,_=self.gru(x);return self.head(h[:,-1]).squeeze(-1)

def build(kind):return RawGRU() if kind=='gru' else SalienceNet(kind)
def model_inputs(data,kind):
    v,k,q,m,noisy,y=data;hint=m if kind=='perfect' else (noisy if kind=='noisy' else torch.zeros_like(m));return v,k,q,hint
@torch.no_grad()
def accuracy(model,data,kind,force_score=None):
    inp=model_inputs(data,kind);logits=[]
    for i in range(0,len(data[0]),512):
        args=tuple(z[i:i+512] for z in inp)
        logits.append(model(*args,force_score=force_score) if force_score is not None and isinstance(model,SalienceNet) else model(*args))
    return ((torch.cat(logits)>0)==data[-1].bool()).float().mean().item()
@torch.no_grad()
def score_diagnostic(model):
    q=torch.eye(4).repeat_interleave(4,0);k=torch.eye(4).repeat(4,1)[:,None,:]
    s=model.learned_scores(k,q).flatten().numpy();y=np.eye(4).flatten().astype(bool)
    ranks=rankdata(s);auc=(ranks[y].sum()-4*5/2)/(4*12)
    return {'raw_auc_16_pairs':float(auc),'orientation_free_auc':float(max(auc,1-auc)),'match_score_mean':float(s[y].mean()),'nonmatch_score_mean':float(s[~y].mean()),'score_matrix':s.reshape(4,4).tolist()}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=100);ap.add_argument('--seeds',type=int,nargs='+',default=[11,22,33]);ap.add_argument('--out',default='results/salience');args=ap.parse_args();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);rows=[];histories={}
    (out/'config.json').write_text(json.dumps(vars(args)|{'variants':VARIANTS,'train_n':3072,'val_n':768,'test_n':4096,'batch':256,'lr':.003,'W_lr':.00015,'supervised_aux_weight':.2,'torch':torch.__version__,'conditions':CONDITIONS},indent=2))
    for seed in args.seeds:
        train=dataset(3072,seed*1000);val=dataset(768,seed*1000+1);tests={c:dataset(4096,seed*1000+10+i,*cfg) for i,(c,cfg) in enumerate(CONDITIONS.items())}
        for kind in VARIANTS:
            torch.manual_seed(seed);model=build(kind);initial=copy.deepcopy(model.state_dict());groups=[{'params':[p for n,p in model.named_parameters() if p.requires_grad and n!='W'],'lr':.003}]
            if isinstance(model,SalienceNet):groups.append({'params':[model.W],'lr':.00015})
            opt=torch.optim.Adam(groups);gen=torch.Generator().manual_seed(seed+5000);best=-1;state=None;best_epoch=0;history=[];start=time.perf_counter();inp=model_inputs(train,kind)
            for epoch in range(args.epochs):
                model.train();loss_sum=0
                for ids in torch.randperm(len(train[0]),generator=gen).split(256):
                    opt.zero_grad(set_to_none=True);batch=tuple(z[ids] for z in inp)
                    if kind=='supervised':
                        logit,score=model(*batch,return_scores=True);loss=F.binary_cross_entropy_with_logits(logit,train[-1][ids])+.2*F.binary_cross_entropy(score,train[3][ids])
                    else:loss=F.binary_cross_entropy_with_logits(model(*batch),train[-1][ids])
                    loss.backward();nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);opt.step();loss_sum+=loss.item()*len(ids)
                model.eval();va=accuracy(model,val,kind);history.append({'epoch':epoch+1,'loss':loss_sum/len(train[0]),'val_accuracy':va})
                if va>best:best=va;state=copy.deepcopy(model.state_dict());best_epoch=epoch+1
            elapsed=time.perf_counter()-start;model.load_state_dict(state);model.eval()
            row={'seed':seed,'kind':kind,'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),'best_epoch':best_epoch,'val_accuracy':best,'train_seconds':elapsed,**{c:accuracy(model,d,kind) for c,d in tests.items()}}
            if isinstance(model,SalienceNet):
                for k in ['B','C','mask']:assert torch.equal(model.state_dict()[k],initial[k])
                if kind in ['learned','supervised']:
                    row.update(score_diagnostic(model));row['score_zero_id']=accuracy(model,tests['id'],kind,force_score=0.);row['score_zero_shift']=accuracy(model,tests['shift'],kind,force_score=0.)
            rows.append(row);histories[f'{kind}_{seed}']=history
            torch.save({'kind':kind,'seed':seed,'state_dict':state},out/f'{kind}_seed{seed}.pt');(out/'metrics.json').write_text(json.dumps(rows,indent=2));(out/'history.json').write_text(json.dumps(histories))
            print(f"Salience {seed}/{kind}: id={row['id']:.4f}, shift={row['shift']:.4f}, delay={row['delay']:.4f}",flush=True)
    summary={k:{c:{'mean':float(np.mean([r[c] for r in rows if r['kind']==k])),'std':float(np.std([r[c] for r in rows if r['kind']==k],ddof=1))} for c in ['id','shift','random','delay','noise','trainable']} for k in VARIANTS};(out/'summary.json').write_text(json.dumps(summary,indent=2));print('SALIENCE COMPLETE',flush=True)
if __name__=='__main__':main()

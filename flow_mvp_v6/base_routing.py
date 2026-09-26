"""先存A/B后给规则：局部读出、动态/静态/无路由的从头训练。"""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

VARIANTS=['dynamic','static','none','rule_blind','dynamic_slow','gru']
CONDITIONS={'id':(10,4,False),'late_rule':(10,6,False),'long':(16,4,False),'reverse_input':(10,4,True)}

def dataset(n,seed,steps=10,arrival=4,reverse=False):
    gen=torch.Generator().manual_seed(seed);ab=torch.randn(n,2,generator=gen);r=torch.randint(2,(n,),generator=gen)
    x=torch.zeros(n,steps,2);x[:,1 if reverse else 0,0]=ab[:,0];x[:,0 if reverse else 1,1]=ab[:,1]
    meta=torch.zeros(n,steps,3);meta[:,:,2]=torch.linspace(0,1,steps)
    meta[:,arrival:,0]=1;meta[:,arrival:,1]=(r.float()*2-1)[:,None]
    target=torch.where(r[:,None].bool(),ab.flip(1),ab)>0
    return x,meta,target.float(),ab,r

def counterfactual(data):
    x,meta,y,ab,r=data;other=meta.clone();other[:,:,1]*=-1;return x,other,y.flip(1),ab,1-r

class RoutedNet(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        allowed=torch.tensor([[1,0,1,1],[0,1,1,1],[0,0,1,0],[0,0,0,1]],dtype=torch.float32)
        mask=(torch.rand(64,64)<.3).float()*allowed.repeat_interleave(16,0).repeat_interleave(16,1)
        w=torch.randn(64,64)*mask
        self.register_buffer('mask',mask);self.W=nn.Parameter(.9*w/torch.linalg.matrix_norm(w,2),requires_grad=kind=='dynamic_slow')
        B=torch.zeros(2,64);B[0,:16]=torch.randn(16)*.7;B[1,16:32]=torch.randn(16)*.7;self.register_buffer('B',B)
        self.write=nn.Sequential(nn.Linear(3,8),nn.Tanh(),nn.Linear(8,2))
        self.hold=nn.Linear(3,4);nn.init.zeros_(self.hold.weight);nn.init.zeros_(self.hold.bias)
        self.route=nn.Sequential(nn.Linear(7,16),nn.Tanh(),nn.Linear(16,16))
        self.static=nn.Parameter(torch.zeros(4,4),requires_grad=kind=='static')
        self.headX=nn.Linear(16,1);self.headY=nn.Linear(16,1)
        with torch.no_grad():self.headX.weight.mul_(5);self.headY.weight.mul_(5)
        for p in self.route.parameters():p.requires_grad_(kind in ['dynamic','rule_blind','dynamic_slow'])
    def forward(self,x,meta,trace=False,gate_override=None):
        h=x.new_zeros(len(x),4,16);gs=[];hs=[]
        w=(self.W*self.mask).reshape(4,16,4,16)
        for t in range(x.shape[1]):
            ctl=meta[:,t];p=self.write(ctl).sigmoid();a=.2*self.hold(ctl).sigmoid()
            if self.kind=='static':g=self.static.sigmoid()[None].expand(len(x),-1,-1)
            elif self.kind=='none':g=x.new_ones(len(x),4,4)
            else:
                direct=ctl.clone()
                if self.kind=='rule_blind':direct[:,1]=0
                g=self.route(torch.cat([direct,h.mean(-1)],1)).sigmoid().reshape(-1,4,4)
            if gate_override is not None:g=gate_override[None].expand_as(g)
            projection=torch.einsum('bsi,sidj->bsdj',h,w)
            recurrent=(projection*g[:,:,:,None]).sum(1)
            candidate=torch.tanh(recurrent+((x[:,t]*p)@self.B).reshape(-1,4,16))
            h=(1-a[:,:,None])*h+a[:,:,None]*candidate
            if trace:gs.append(g.detach());hs.append(h.detach())
        logits=torch.cat([self.headX(h[:,2]),self.headY(h[:,3])],1)
        return (logits,torch.stack(gs,1),torch.stack(hs,1)) if trace else logits

class RoutingGRU(nn.Module):
    def __init__(self):
        super().__init__();self.gru=nn.GRU(5,32,batch_first=True);self.head=nn.Linear(32,2)
    def forward(self,x,meta):
        h,_=self.gru(torch.cat([x,meta],-1));return self.head(h[:,-1])

def build(kind):return RoutingGRU() if kind=='gru' else RoutedNet(kind)
@torch.no_grad()
def predictions(model,data,gate_override=None):
    x,meta=data[:2];out=[]
    for i in range(0,len(x),512):
        if isinstance(model,RoutedNet):out.append(model(x[i:i+512],meta[i:i+512],gate_override=gate_override))
        else:out.append(model(x[i:i+512],meta[i:i+512]))
    return torch.cat(out)>0
@torch.no_grad()
def evaluate(model,data):
    pred=predictions(model,data);truth=data[2].bool();correct=pred==truth;opposite=(data[3][:,0]*data[3][:,1])<0
    return {'bit_accuracy':correct.float().mean().item(),'pair_accuracy':correct.all(1).float().mean().item(),'opposite_pair_accuracy':correct[opposite].all(1).float().mean().item()}
@torch.no_grad()
def routing_diagnostics(model,data,train):
    sample=tuple(z[:256] for z in data);cf=counterfactual(sample)
    logit,g,h=model(*sample[:2],trace=True);other,g2,h2=model(*cf[:2],trace=True)
    assert torch.equal(h[:,:4],h2[:,:4]),'Future rule leaked before arrival'
    counts=model.mask.reshape(4,16,4,16).sum((1,3));weights=counts/counts.sum();safe=g.clamp(1e-7,1-1e-7)
    entropy=-(safe*safe.log2()+(1-safe)*(1-safe).log2())
    out={'weighted_gate_entropy_bits':float((entropy*weights).sum((-2,-1)).mean()),'paired_rule_gate_distance':float(((g[:,4:]-g2[:,4:]).abs()*weights).sum((-2,-1)).mean()),'before_rule_state_difference':float((h[:,:4]-h2[:,:4]).abs().max())}
    for threshold in [.1,.5,.9]:out[f'edge_fraction_gate_gt_{threshold}']=float(((g>threshold).float()*weights).sum((-2,-1)).mean())
    # 同一A/B两种规则；按R重新整理，观察源到目的块的门而不是仅准确率。
    gzero=torch.where(sample[4][:,None,None,None].bool(),g2,g);gone=torch.where(sample[4][:,None,None,None].bool(),g,g2)
    out['rule0_post_gate_matrix']=gzero[:,4:].mean((0,1)).tolist();out['rule1_post_gate_matrix']=gone[:,4:].mean((0,1)).tolist()
    correct=(logit>0)==sample[2].bool();correct2=(other>0)==cf[2].bool();opposite=(sample[3][:,0]*sample[3][:,1])<0
    out['both_rules_correct_opposite']=float((correct.all(1)&correct2.all(1))[opposite].float().mean())
    tr=tuple(z[:512] for z in train);_,trg,_=model(*tr[:2],trace=True);mean_gate=trg.mean((0,1))
    override=predictions(model,data,gate_override=mean_gate)
    out['global_static_intervention_pair_accuracy']=float((override==data[2].bool()).all(1).float().mean())
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=100);ap.add_argument('--seeds',type=int,nargs='+',default=[11,22,33]);ap.add_argument('--out',default='results/routing');args=ap.parse_args();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);rows=[];history={}
    (out/'config.json').write_text(json.dumps(vars(args)|{'variants':VARIANTS,'train_n':3072,'val_n':768,'test_n':4096,'batch':256,'lr':.003,'slow_W_lr':.00015,'conditions':CONDITIONS,'torch':torch.__version__},indent=2))
    for seed in args.seeds:
        train=dataset(3072,seed*1000);val=dataset(768,seed*1000+1);tests={c:dataset(4096,seed*1000+10+i,*cfg) for i,(c,cfg) in enumerate(CONDITIONS.items())}
        for kind in VARIANTS:
            torch.manual_seed(seed);model=build(kind);initial=copy.deepcopy(model.state_dict());groups=[{'params':[p for n,p in model.named_parameters() if p.requires_grad and n!='W'],'lr':.003}]
            if kind=='dynamic_slow':groups.append({'params':[model.W],'lr':.00015})
            opt=torch.optim.Adam(groups);gen=torch.Generator().manual_seed(seed+5000);best=-1;state=None;best_epoch=0;hh=[];start=time.perf_counter()
            for epoch in range(args.epochs):
                model.train();loss_sum=0
                for ids in torch.randperm(len(train[0]),generator=gen).split(256):
                    opt.zero_grad(set_to_none=True);loss=F.binary_cross_entropy_with_logits(model(train[0][ids],train[1][ids]),train[2][ids]);loss.backward();nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);opt.step();loss_sum+=loss.item()*len(ids)
                model.eval();va=evaluate(model,val)['pair_accuracy'];hh.append({'epoch':epoch+1,'loss':loss_sum/len(train[0]),'val_pair_accuracy':va})
                if va>best:best=va;state=copy.deepcopy(model.state_dict());best_epoch=epoch+1
            elapsed=time.perf_counter()-start;model.load_state_dict(state);model.eval();r={'seed':seed,'kind':kind,'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),'val_pair_accuracy':best,'best_epoch':best_epoch,'train_seconds':elapsed}
            for c,d in tests.items():r.update({f'{c}_{k}':v for k,v in evaluate(model,d).items()})
            if isinstance(model,RoutedNet):
                for k in ['B','mask']+([] if kind=='dynamic_slow' else ['W']):assert torch.equal(model.state_dict()[k],initial[k])
                r.update(routing_diagnostics(model,tests['id'],train))
            rows.append(r);history[f'{kind}_{seed}']=hh;torch.save({'kind':kind,'seed':seed,'state_dict':state},out/f'{kind}_seed{seed}.pt')
            (out/'metrics.json').write_text(json.dumps(rows,indent=2));(out/'history.json').write_text(json.dumps(history))
            print(f"Routing {seed}/{kind}: pair={r['id_pair_accuracy']:.4f}, opposite={r['id_opposite_pair_accuracy']:.4f}, long={r['long_pair_accuracy']:.4f}",flush=True)
    keys=[f'{c}_{k}' for c in CONDITIONS for k in ['bit_accuracy','pair_accuracy','opposite_pair_accuracy']]+['trainable']
    summary={kind:{k:{'mean':float(np.mean([r[k] for r in rows if r['kind']==kind])),'std':float(np.std([r[k] for r in rows if r['kind']==kind],ddof=1))} for k in keys} for kind in VARIANTS};(out/'summary.json').write_text(json.dumps(summary,indent=2));print('ROUTING COMPLETE',flush=True)
if __name__=='__main__':main()

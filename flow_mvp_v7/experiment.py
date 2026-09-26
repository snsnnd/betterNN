import argparse,copy,json,math,time,os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
torch.set_num_threads(int(os.environ.get("FLOW_THREADS","1")));torch.use_deterministic_algorithms(True)
SIZES=[64,128,256,512,1024];SEEDS=[11,22,33,44,55]
EPOCHS=30

def data(seed,n,task,steps=20):
 g=torch.Generator().manual_seed(seed);ab=torch.randn(n,2,generator=g);r=torch.randint(2,(n,),generator=g)
 x=torch.zeros(n,steps,2);x[:,0,0]=ab[:,0];x[:,1,1]=ab[:,1]
 meta=torch.zeros(n,steps,7);meta[:,:,2]=torch.linspace(0,1,steps);meta[:,4:,0]=1;meta[:,4:,1]=(2*r.float()-1)[:,None];meta[:,:,3+task]=1
 idx=torch.tensor([[[0,1],[1,0]],[[0,0],[1,1]],[[1,1],[0,0]],[[1,0],[0,1]]])[task,r]
 return x,meta,(ab.gather(1,idx)>0).float()

class Net(nn.Module):
 def __init__(self,n,mode):
  super().__init__();self.n=n;self.m=n//4;self.mode=mode;self.window=5
  allowed=torch.tensor([[1,0,1,1],[0,1,1,1],[0,0,1,0],[0,0,0,1.]],dtype=torch.float32)
  mask=(torch.rand(n,n)<.3).float()*allowed.repeat_interleave(self.m,0).repeat_interleave(self.m,1)
  w=torch.randn(n,n)*mask;self.register_buffer('mask',mask);self.W=nn.Parameter(.9*w/torch.linalg.matrix_norm(w,2))
  b=torch.zeros(2,n);b[0,:self.m]=torch.randn(self.m)*.7;b[1,self.m:2*self.m]=torch.randn(self.m)*.7;self.register_buffer('B',b)
  self.write=nn.Sequential(nn.Linear(7,8),nn.Tanh(),nn.Linear(8,2));self.hold=nn.Linear(7,4);nn.init.zeros_(self.hold.weight);nn.init.zeros_(self.hold.bias)
  self.route=nn.Sequential(nn.Linear(11,16),nn.Tanh(),nn.Linear(16,16));self.headX=nn.Linear(self.m,1);self.headY=nn.Linear(self.m,1)
  with torch.no_grad():self.headX.weight.mul_(5);self.headY.weight.mul_(5)
  self.selector=nn.Parameter(torch.randn(4,4,self.m)*.01,requires_grad=mode=='topk' and n>64)
 def masks(self,task):
  z=self.selector[task];soft=16*torch.softmax(z,-1);hard=torch.zeros_like(z).scatter_(-1,z.topk(16,dim=-1).indices,1.)
  if self.mode=='dense' or self.n==64:return torch.ones_like(z),torch.ones_like(z)
  return hard+(soft-soft.detach()),hard
 def forward(self,x,meta,trace=False):
  # Single task in each batch; explicit context is observed from t=0 in both arms.
  task=int(meta[0,0,3:].argmax());assert bool((meta[:,0,3:].argmax(-1)==task).all())
  sel,hard=self.masks(task);h=x.new_zeros(len(x),4,self.m);w=(self.W*self.mask).reshape(4,self.m,4,self.m);gs=[];hs=[]
  scale=math.sqrt(self.n/64) if self.mode=='topk' else 1.
  for t in range(x.shape[1]):
   if self.training and t and t%self.window==0:h=h.detach()
   ctl=meta[:,t];p=self.write(ctl).sigmoid();a=.2*self.hold(ctl).sigmoid()
   # Active-normalized pool prevents larger inactive population diluting state summary.
   pooled=h.sum(-1)/(16 if self.mode=='topk' else self.m)
   gate=self.route(torch.cat([ctl,pooled],1)).sigmoid().reshape(-1,4,4)
   rec=(torch.einsum('bsi,sidj->bsdj',h,w)*gate[:,:,:,None]).sum(1)*scale
   cand=torch.tanh(rec+((x[:,t]*p)@self.B).reshape(-1,4,self.m))
   h=((1-a[:,:,None])*h+a[:,:,None]*cand)*sel[None]
   if trace:gs.append(gate.detach());hs.append(h.detach())
  out=torch.cat([F.linear(h[:,2]*scale,self.headX.weight,self.headX.bias),F.linear(h[:,3]*scale,self.headY.weight,self.headY.bias)],1)
  return (out,torch.stack(gs,1),torch.stack(hs,1),hard.detach()) if trace else out

@torch.no_grad()
def evaluate(m,d):
 m.eval();correct=0
 for ids in torch.arange(len(d[0])).split(128):correct+=int(((m(d[0][ids],d[1][ids])>0)==d[2][ids].bool()).all(1).sum())
 return correct/len(d[0])

@torch.no_grad()
def diagnostics(m,seed):
 m.eval();acts=[];gates=[];supports=[];counts=m.mask.reshape(4,m.m,4,m.m).sum((1,3));exists=counts>0
 for task in range(4):
  x,meta,_=data(seed*10000+999,128,task);_,g,h,hard=m(x,meta,trace=True)
  if m.mode=='topk':assert int(hard.sum())==64;assert int((h!=0).sum((-1,-2)).max())<=64
  acts.append(h[:,4:].abs().mean((0,1)).flatten());gates.append(g[:,4:].mean((0,1))[exists]);supports.append(hard.flatten().bool())
 a=torch.stack(acts);gate=torch.stack(gates);support=torch.stack(supports)
 # Fixed fraction activation overlap, distinct from explicit selection overlap.
 kk=max(1,m.n//4);active=torch.zeros_like(a,dtype=torch.bool).scatter_(1,a.topk(kk,dim=1).indices,True)
 def jac(q):return [[float((u&v).sum()/(u|v).sum().clamp_min(1)) for v in q] for u in q]
 mx=a.max(0).values;other=(a.sum(0)-mx)/3;si=(mx-other)/(mx+other+1e-9)
 cosine=F.normalize(gate,dim=1)@F.normalize(gate,dim=1).T
 return {'activation_top25pct_jaccard':jac(active),'selection_jaccard':jac(support),'node_selectivity_mean':float(si.mean()),'route_cosine':cosine.tolist(),'mean_activation':a.tolist(),'mean_route':gate.tolist(),'selected_nodes':support.nonzero().tolist(),'nonzero_activation_nodes':int((a.sum(0)>1e-9).sum())}

def run(n,mode,seed):
 name=f'{mode}_{n}_{seed}';p=OUT/(name+'.json')
 if p.exists():return
 torch.manual_seed(seed);m=Net(n,mode);fixed={k:v.clone() for k,v in m.state_dict().items() if k in ['B','mask']}
 tests=[data(seed*10000+t*100+10,1024,t) for t in range(4)];matrix=[];history={};diag0=diagnostics(m,seed);start=time.perf_counter()
 for task in range(4):
  tr=data(seed*10000+task*100,512,task);val=data(seed*10000+task*100+1,256,task)
  opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.003);gen=torch.Generator().manual_seed(seed+task*100+5000);hh=[]
  for epoch in range(EPOCHS):
   m.train();loss_sum=0
   for ids in torch.randperm(512,generator=gen).split(128):
    opt.zero_grad(set_to_none=True);loss=F.binary_cross_entropy_with_logits(m(tr[0][ids],tr[1][ids]),tr[2][ids]);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();loss_sum+=float(loss.detach())/4
   hh.append({'loss':loss_sum,'validation':evaluate(m,val)})
  history[str(task)]=hh;matrix.append([evaluate(m,d) for d in tests])
  for k,v in fixed.items():assert torch.equal(v,m.state_dict()[k])
  print(name,'stage',task,[round(v,3) for v in matrix[-1]],flush=True)
 diag=diagnostics(m,seed)
 row={'threads':torch.get_num_threads(),'name':name,'n':n,'mode':mode,'seed':seed,'matrix':matrix,'plasticity':float(np.mean([matrix[t][t] for t in range(4)])),'final_mean':float(np.mean(matrix[-1])),'forgetting':float(np.mean([max(matrix[s][t] for s in range(t,4))-matrix[-1][t] for t in range(3)])),'bwt':float(np.mean([matrix[-1][t]-matrix[t][t] for t in range(3)])),'diag_initial':diag0,'diag_final':diag,'history':history,'seconds':time.perf_counter()-start,'trainable':sum(p.numel() for p in m.parameters() if p.requires_grad),'active_budget':64 if mode=='topk' else n}
 torch.save({'state':m.state_dict(),'n':n,'mode':mode,'seed':seed},OUT/(name+'.pt'));p.write_text(json.dumps(row,indent=2))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--sizes',nargs='+',type=int,default=SIZES);ap.add_argument('--seeds',nargs='+',type=int,default=SEEDS);ap.add_argument('--modes',nargs='+',default=['dense','topk']);args=ap.parse_args()
 (ROOT/'config.json').write_text(json.dumps({'sizes':SIZES,'seeds':SEEDS,'modes':['dense','topk'],'epochs_per_task':EPOCHS,'tasks':4,'train':512,'val':256,'test':1024,'batch':128,'bptt_window':5,'lr':.003,'task_identity':'visible from t0','rule':'visible from t4','selection':'final epoch','replay':False,'torch':torch.__version__},indent=2))
 for n in args.sizes:
  for seed in args.seeds:
   for mode in args.modes:run(n,mode,seed)
if __name__=='__main__':main()

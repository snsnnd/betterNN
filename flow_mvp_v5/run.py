import copy,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from base_routing import RoutedNet,RoutingGRU,dataset,evaluate

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
OUT=Path(__file__).parent/'results';OUT.mkdir(exist_ok=True)
# Identical recurrence as v4, with explicit boundaries; eval never truncates.
import inspect
src=inspect.getsource(RoutedNet.forward)
src=src.replace('    def forward','def forward',1)
lines=src.splitlines();src='\n'.join([lines[0]]+[s[4:] for s in lines[1:]])
src=src.replace('ctl=meta[:,t];','if self.training and self.window and t and t % self.window == 0: h=h.detach()\n        ctl=meta[:,t];')
space={'torch':torch};exec(src,space)
class Flow(RoutedNet):
    forward=space['forward']
    def __init__(self,kind,window):
        super().__init__('dynamic_slow' if kind!='frozen' else 'dynamic');self.window=window
class GRU(RoutingGRU):
    def __init__(self,window):super().__init__();self.window=window
    def forward(self,x,meta):
        z=torch.cat([x,meta],-1);h=None
        k=self.window if self.training and self.window else z.shape[1]
        for start in range(0,z.shape[1],k):
            out,h=self.gru(z[:,start:start+k],h)
            if start+k<z.shape[1]:h=h.detach()
        return self.head(out[:,-1])
def build(kind,window):return GRU(window) if kind=='gru' else Flow(kind,window)
def data(seed,n,steps=20):return dataset(n,seed,steps,4)
CONFIG={'seeds':[11,22,33],'kinds':['frozen','slow','joint','gru'],'windows':[0,10,5],'epochs':60,'train_n':1024,'val_n':512,'test_n':2048,'steps':20,'rule_arrival':4,'batch':256,'lr':.003,'slow_lr':.00015,'loss':'final output only','window_zero':'full BPTT','selection':'ID validation pair accuracy','torch':torch.__version__}
(OUT/'config.json').write_text(json.dumps(CONFIG,indent=2))
def main():
    rows=[]
    for seed in CONFIG['seeds']:
        train=data(seed*1000,1024);val=data(seed*1000+1,512)
        tests={'id':data(seed*1000+10,2048),'long':data(seed*1000+11,2048,40)}
        for kind in CONFIG['kinds']:
          for window in CONFIG['windows']:
            name=f'{kind}_{seed}_{window}';torch.manual_seed(seed);model=build(kind,window)
            initial=copy.deepcopy(model.state_dict())
            groups=[{'params':[p for n,p in model.named_parameters() if p.requires_grad and n!='W'],'lr':.003}]
            if kind in ['slow','joint']:groups.append({'params':[model.W],'lr':.00015 if kind=='slow' else .003})
            opt=torch.optim.Adam(groups);gen=torch.Generator().manual_seed(seed+5000)
            best=-1;hist=[];start=time.perf_counter()
            # Causal gradient test: earlier input must have zero gradient under truncation.
            model.train();xx=train[0][:16].clone().requires_grad_();loss=F.binary_cross_entropy_with_logits(model(xx,train[1][:16]),train[2][:16]);loss.backward()
            gradient=float(xx.grad[:,:2].abs().sum());assert (gradient==0) if window else (gradient>0)
            for epoch in range(CONFIG['epochs']):
                model.train();total=0
                for ids in torch.randperm(1024,generator=gen).split(256):
                    opt.zero_grad(set_to_none=True);loss=F.binary_cross_entropy_with_logits(model(train[0][ids],train[1][ids]),train[2][ids]);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=float(loss)*len(ids)
                model.eval();score=evaluate(model,val)['pair_accuracy'];hist.append([total/1024,score])
                if score>best:best=score;state=copy.deepcopy(model.state_dict());best_epoch=epoch+1
            model.load_state_dict(state);model.eval()
            r={'name':name,'seed':seed,'kind':kind,'window':window,'best_epoch':best_epoch,'val':best,'seconds':time.perf_counter()-start,'initial_early_input_gradient':gradient,'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad)}
            for cond,d in tests.items():r[cond]=evaluate(model,d)
            if kind!='gru':
                for key in ['B','mask']+(['W'] if kind=='frozen' else []):assert torch.equal(initial[key],state[key])
            torch.save({'state':state,'row':r,'history':hist},OUT/(name+'.pt'))
            rows.append(r);(OUT/'metrics.json').write_text(json.dumps(rows,indent=2))
            print(name,round(r['id']['pair_accuracy'],4),round(r['long']['pair_accuracy'],4),flush=True)
    (OUT/'DONE').write_text('36 training runs completed\n')

if __name__=="__main__":main()

import copy,json
import torch
from torch import nn
from torch.nn import functional as F
from window_models import Flow,data,evaluate,OUT

def build(kind,window):
    m=Flow('joint',window);m.kind=kind
    m.static.requires_grad_(kind=='static')
    for p in m.route.parameters():p.requires_grad_(False)
    return m

def main():
 rows=[]
 for seed in [11,22,33]:
  train=data(seed*1000,1024);val=data(seed*1000+1,512)
  for kind in ['static','none']:
   for window in [0,5]:
    torch.manual_seed(seed);m=build(kind,window);opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.003);gen=torch.Generator().manual_seed(seed+5000);best=-1;history=[]
    for epoch in range(60):
     m.train()
     for ids in torch.randperm(1024,generator=gen).split(256):
      opt.zero_grad(set_to_none=True);loss=F.binary_cross_entropy_with_logits(m(train[0][ids],train[1][ids]),train[2][ids]);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
     m.eval();score=evaluate(m,val)['pair_accuracy'];history.append(score)
     if score>best:best=score;state=copy.deepcopy(m.state_dict());best_epoch=epoch+1
    m.load_state_dict(state);m.eval();r={'seed':seed,'kind':kind,'window':window,'best_epoch':best_epoch,'val':best,'id':evaluate(m,data(seed*1000+10,2048)),'long':evaluate(m,data(seed*1000+11,2048,40))}
    name=f'{kind}_{seed}_{window}';torch.save({'state':state,'row':r,'history':history},OUT/(name+'.pt'));rows.append(r);(OUT/'metrics.json').write_text(json.dumps(rows,indent=2));print(name,r['id']['pair_accuracy'],flush=True)
 (OUT/'DONE').write_text('12 retrained controls completed')
if __name__=='__main__':main()

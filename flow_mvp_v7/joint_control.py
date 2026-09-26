import json
import torch
from torch import nn
from torch.nn import functional as F
from experiment import ROOT,Net,data,evaluate,SEEDS,EPOCHS
out=ROOT/'joint_control';out.mkdir(exist_ok=True)
for seed in SEEDS:
 torch.manual_seed(seed);m=Net(64,'dense');train=[data(seed*10000+t*100,512,t) for t in range(4)];tests=[data(seed*10000+t*100+10,1024,t) for t in range(4)];gen=torch.Generator().manual_seed(seed+99000);updates=0
 for epoch in range(EPOCHS):
  batches=[list(torch.randperm(512,generator=gen).split(128)) for _ in range(4)]
  for b in range(4):
   for t in torch.randperm(4,generator=gen).tolist():
    if updates%120==0:opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.003)
    m.train();ids=batches[t][b];d=train[t];opt.zero_grad(set_to_none=True);loss=F.binary_cross_entropy_with_logits(m(d[0][ids],d[1][ids]),d[2][ids]);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();updates+=1
 scores=[evaluate(m,d) for d in tests];row={'seed':seed,'scores':scores,'updates':updates,'mean':sum(scores)/4}
 torch.save(m.state_dict(),out/f'{seed}.pt');(out/f'{seed}.json').write_text(json.dumps(row,indent=2));print(seed,scores,flush=True)

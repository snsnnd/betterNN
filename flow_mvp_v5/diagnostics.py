import json
import torch
from torch.nn import functional as F
from run import build,data,OUT
rows=json.loads((OUT/'metrics.json').read_text());diag=[]
for r in rows:
    torch.manual_seed(r['seed']);m=build(r['kind'],r['window']);initial={k:v.clone() for k,v in m.state_dict().items()};m.load_state_dict(torch.load(OUT/(r['name']+'.pt'),weights_only=False)['state'])
    m.train();x,meta,y,*_=data(r['seed']*1000,128)
    loss=F.binary_cross_entropy_with_logits(m(x,meta),y);loss.backward()
    d={'name':r['name'],'gradient_norms':{},'parameter_change_l2':{}}
    for name,p in m.named_parameters():
        group=name.split('.')[0];d['gradient_norms'][group]=d['gradient_norms'].get(group,0)+(float(p.grad.detach().square().sum()) if p.grad is not None else 0)
        d['parameter_change_l2'][group]=d['parameter_change_l2'].get(group,0)+float((p.detach()-initial[name]).square().sum())
    for category in ['gradient_norms','parameter_change_l2']:d[category]={k:v**.5 for k,v in d[category].items()}
    if r['kind']!='gru':
        m.eval()
        with torch.no_grad():_,g,_=m(x,meta,trace=True)
        counts=m.mask.reshape(4,16,4,16).sum((1,3));weights=counts/counts.sum()
        d['edge_weighted_gate_gt_half']=float(((g>.5).float()*weights).sum((-1,-2)).mean())
    diag.append(d)
(OUT/'diagnostics.json').write_text(json.dumps(diag,indent=2))
print('Gradient/weight-change/route-usage diagnostics saved for',len(diag),'models')

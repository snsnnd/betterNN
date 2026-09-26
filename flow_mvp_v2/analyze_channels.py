"""针对随机种子波动，分任务通道诊断，不更新或选择模型。"""
import json
from pathlib import Path
import torch
from experiment import build,dataset

torch.set_num_threads(2);rows=[]
for seed in [11,22,33]:
 model=build('marker_memory');ck=torch.load(f'results/marker_memory_seed{seed}.pt',weights_only=True,map_location='cpu');model.load_state_dict(ck['state_dict']);model.eval()
 x,q,m,y=dataset(4096,seed*1000+10)
 with torch.no_grad():logit,p,g,a=model(x,q,m,trace=True)
 for channel in range(4):
  ids=q[:,channel].bool();valid=m[ids].bool();injection=p[ids,:,channel]
  rows.append({'seed':seed,'channel':channel,'n':int(ids.sum()),'accuracy':((logit[ids]>0)==y[ids].bool()).float().mean().item(),'valid_target_injection':injection[valid].mean().item()})
Path('results/channel_diagnostics.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))

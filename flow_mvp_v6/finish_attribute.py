import json
import numpy as np
import torch
from attribute import build
from window_models import OUT,data,evaluate
rows=json.loads((OUT/'metrics.json').read_text());assert len(rows)==12
old=[r for r in json.loads((OUT/'dynamic_v5_metrics.json').read_text()) if r['kind']=='joint' and r['window'] in [0,5]]
for r in rows:
 torch.manual_seed(r['seed']);m=build(r['kind'],r['window']);before={k:v.clone() for k,v in m.state_dict().items()};m.load_state_dict(torch.load(OUT/f"{r['kind']}_{r['seed']}_{r['window']}.pt",weights_only=False)['state']);m.eval()
 for i,(c,n) in enumerate([('id',20),('long',40)]):assert evaluate(m,data(r['seed']*1000+10+i,2048,n))==r[c]
 for key in ['B','mask']:assert torch.equal(before[key],m.state_dict()[key])
lines=['# 补充实验：99%需要动态Router吗？','','收到用户粘贴意见后新增12次配对训练，正常更新W，其他条件与第五轮一致。动态组复用第五轮已核验的成绩，静态/全开从头训练。','','| Router | 梯度窗口 | 20步全对 | 40步全对 |','|---|---|---|---|']
for kind in ['joint','static','none']:
 for w in [0,5]:
  rr=[r for r in rows+old if r['kind']==kind and r['window']==w];vals=[]
  for c in ['id','long']:
   a=[r[c]['pair_accuracy']*100 for r in rr];vals.append(f'{np.mean(a):.2f} ± {np.std(a,ddof=1):.2f}%')
  lines.append(f'| {kind} | {w or "完整"} | '+' | '.join(vals)+' |')
lines+=['','joint为动态Router，static为16个可训练常数门，none为通路全开；三组Write/Hold和读出均可学习，均可观察相同规则。门类型改变了参数量，不能完全隔离额外容量。静态/全开仍可能利用Write/Hold实施条件计算，因此成功不意味着网络没有任何动态控制。','','每组3种子，60轮，1024/512/2048样本，验证选模型；20步训练40步测试。此实验能检验当前结构与预算下Router增益，不能证明其不可替代，或等价于一般RNN。尚未证明学习过程局部可解释；短窗口共享参数和持续可见规则均可能起作用。','','12个新检查点重载，24组测试指标一致；B和mask保持固定。']
(OUT.parent/'ATTRIBUTION_REPORT.md').write_text('\n'.join(lines));print('\n'.join(lines))

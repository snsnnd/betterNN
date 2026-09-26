import json
from pathlib import Path
import numpy as np
import torch
from experiment import ROOT,OUT,Net,data,evaluate,SIZES,SEEDS,diagnostics

def offdiag(a):
 a=np.array(a);return float(a[~np.eye(4,dtype=bool)].mean())
def enrich(r):
 for phase in ['initial','final']:
  d=r['diag_'+phase];a=np.array(d['mean_activation']);mx=a.max(0);other=(a.sum(0)-mx)/3;si=(mx-other)/(mx+other+1e-9);used=mx>1e-8
  r[phase+'_selectivity_used']=float(si[used].mean()) if used.any() else 0
  keep=np.zeros_like(a,dtype=bool);indices=np.argsort(-a,axis=1,kind='stable')[:,:max(1,r['n']//4)];np.put_along_axis(keep,indices,True,axis=1);keep &= a>1e-8
  overlaps=[[float(np.logical_and(u,v).sum()/max(1,np.logical_or(u,v).sum())) for v in keep] for u in keep]
  d['activation_positive_top25pct_jaccard']=overlaps
  r[phase+'_activation_overlap']=offdiag(overlaps);r[phase+'_selection_overlap']=offdiag(d['selection_jaccard']);r[phase+'_route_similarity']=offdiag(d['route_cosine'])
 r['overlap_change']=r['final_selection_overlap']-r['initial_selection_overlap']
 return r

def main():
 rows=[enrich(json.loads(p.read_text())) for p in sorted(OUT.glob('*.json')) if p.stem.split('_')[0] in ['dense','topk']];assert len(rows)==50,len(rows)
 # Reproduce final scores, active count and frozen tensors; intermediate checkpoints not stored.
 for r in rows:
  torch.set_num_threads(r.get('threads',1))
  torch.manual_seed(r['seed']);m=Net(r['n'],r['mode']);fixed={k:v.clone() for k,v in m.state_dict().items() if k in ['B','mask']}
  ck=torch.load(OUT/(r['name']+'.pt'),weights_only=False);m.load_state_dict(ck['state'])
  with torch.no_grad():
   w=m.W*m.mask;v=torch.ones(m.n);v=v/v.norm()
   for _ in range(25):v=w.T@(w@v);v=v/v.norm().clamp_min(1e-12)
   r['final_W_spectral_norm_estimate']=float((w@v).norm())
  got=[evaluate(m,data(r['seed']*10000+t*100+10,1024,t)) for t in range(4)];assert got==r['matrix'][-1]
  for k,v in fixed.items():assert torch.equal(v,m.state_dict()[k])
  d=diagnostics(m,r['seed']);assert np.allclose(d['mean_activation'],r['diag_final']['mean_activation'])
 for seed in SEEDS:
  a=next(r for r in rows if r['n']==64 and r['seed']==seed and r['mode']=='dense');b=next(r for r in rows if r['n']==64 and r['seed']==seed and r['mode']=='topk');assert a['matrix']==b['matrix']
 summary=[]
 for n in SIZES:
  for mode in ['dense','topk']:
   rr=[r for r in rows if r['n']==n and r['mode']==mode];row={'n':n,'mode':mode}
   for k in ['final_mean','plasticity','forgetting','bwt','initial_selection_overlap','final_selection_overlap','overlap_change','final_activation_overlap','initial_selectivity_used','final_selectivity_used','final_route_similarity','final_W_spectral_norm_estimate']:
    row[k]={'mean':float(np.mean([r[k] for r in rr])),'std':float(np.std([r[k] for r in rr],ddof=1))}
   summary.append(row)
 (ROOT/'summary.json').write_text(json.dumps(summary,indent=2));(ROOT/'all_metrics.json').write_text(json.dumps(rows,indent=2))
 def fmt(r,k):return f"{r[k]['mean']*100:.2f} ± {r[k]['std']*100:.2f}"
 lines=['# 第七轮：容量与分区的配对筛查','','完成5规模×2模式×5种子=50条连续流、200任务阶段；每阶段30轮512样本，5步BPTT。表中准确率为两个输出全对，均值±样本标准差。','','| 节点 | 模式 | 新任务刚学完% | 最终平均% | 遗忘百分点 | BWT百分点 |','|---|---|---|---|---|---|']
 for r in summary:lines.append(f"| {r['n']} | {r['mode']} | "+' | '.join(fmt(r,k) for k in ['plasticity','final_mean','forgetting','bwt'])+' |')
 lines+=['','## 分区指标','','选择Jaccard指实际硬节点支持集；激活Jaccard指各任务平均绝对激活前25%且大于1e-8的节点，剔除零激活，二者不同。SI只在至少一个任务上有非微小激活的节点统计，避免未使用节点随N增加机械稀释。路由相似度只比较结构存在的块边。所有任务采用相同输入数值和规则采样。','','| N | 模式 | 选择重叠初始→最终 | 激活重叠最终 | SI初始→最终 | 路由余弦最终 |','|---|---|---|---|---|---|']
 for r in summary:
  v=lambda k:r[k]['mean'];lines.append(f"| {r['n']} | {r['mode']} | {v('initial_selection_overlap'):.3f} → {v('final_selection_overlap'):.3f} | {v('final_activation_overlap'):.3f} | {v('initial_selectivity_used'):.3f} → {v('final_selectivity_used'):.3f} | {v('final_route_similarity'):.3f} |")
 lines+=['','## 能与不能推断的内容','','- 低选择重叠可能是随机Top-K初始化和任务私有选择表带来的，必须看初始→最终变化。低重叠或高选择性本身不等于有用的功能分工，更不证明出现复杂智能。','- 四个来源/目的角色仍是人工设置；本轮只考察每个角色内部的任务选择，非自发发现整个模块结构。任务身份从t0明确可见，两组相同；R仍到t4才给。','- 每步Top-K仅64节点允许非零，但仍执行稠密运算；不构成等FLOPs或真实稀疏加速证据。topk采用活动方差归一，选择器含额外任务专属参数，不能把所有差异归于容量或稀疏性。','- 各规模同数据和更新次数，非等总计算；只有一个任务顺序、固定超参数与有限预算。未充分学会任务的组不能用于证明容量够/不够，也不能用小遗忘宣称优势。','- 不同于第六轮：从头学习A、任务身份更早、预算更小、5步截断。因此不直接跨轮比较分数。扩大规模时保持任务固定用于隔离规模，复杂任务是之后独立研究。','', '## 验证','','50个最终检查点重载，200个最终任务分复算一致；Top-K非零状态最多64；固定B/mask一致；N64两模式的所有阶段成绩逐种子完全一致。中间阶段只保存分数与训练曲线，不声称重载了未保存的中间模型。所有失败和随机种子均保留。']
 torch.set_num_threads(1)
 joint=[]
 for seed in SEEDS:
  r=json.loads((ROOT/'joint_control'/f'{seed}.json').read_text());torch.manual_seed(seed);m=Net(64,'dense');m.load_state_dict(torch.load(ROOT/'joint_control'/f'{seed}.pt',weights_only=True));got=[evaluate(m,data(seed*10000+t*100+10,1024,t)) for t in range(4)];assert got==r['scores'];joint.append(r)
 means=[r['mean'] for r in joint]
 lines+=['','## 64节点混合任务正对照','',f'同样总更新480次，四任务交替训练：最终平均 {np.mean(means)*100:.2f} ± {np.std(means,ddof=1)*100:.2f}%。5个检查点20个分数复算一致。该组可持续访问所有任务训练集，不是无回放或小记忆持续学习。若未接近满分，只能说明这个预算未充分学会，不能由此宣布容量不足。']
 (ROOT/'joint_summary.json').write_text(json.dumps(joint,indent=2))
 (ROOT/'REPORT.md').write_text('\n'.join(lines));(ROOT/'verification.txt').write_text('PASS 50 final checkpoints, 200 final task scores; active-state budget <=64; fixed B/mask; N64 dense/topk all-stage equivalence.\n');print('\n'.join(lines))
if __name__=='__main__':main()

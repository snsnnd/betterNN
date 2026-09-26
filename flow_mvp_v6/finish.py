import json
import numpy as np
import torch
from run import OUT,ROOT,initial,data,scores
from base_routing import RoutedNet,RoutingGRU,dataset
rows=json.loads((OUT/'metrics.json').read_text());controls=json.loads((OUT/'controls.json').read_text());assert len(rows)==12 and len(controls)==9
# Zero extension preserves pretrained function and no task identity leaks before rule.
for seed in [11,22,33]:
 for kind in ['joint','gru']:
    old=RoutingGRU() if kind=='gru' else RoutedNet('dynamic_slow')
    old.load_state_dict(torch.load(ROOT/'initial'/f'{kind}_{seed}_0.pt',weights_only=False)['state']);old.eval();new=initial(kind,seed);new.eval()
    x,meta=dataset(32,17,20,4)[:2];dd=data(17,32,0)
    with torch.no_grad():assert torch.allclose(old(x,meta),new(*dd[:2]),atol=1e-6)
 for t in range(4):
    d=data(19,32,t);assert torch.equal(d[1][:,:4,3:],torch.zeros_like(d[1][:,:4,3:]))
for row in rows:
 for stage in [1,2,3]:
    m=initial(row['kind'],row['seed']);ck=torch.load(OUT/f"{row['kind']}_{row['seed']}_stage{stage}.pt",weights_only=False);m.load_state_dict(ck['state']);assert scores(m,row['seed'])==row['matrix'][stage]
for c in controls:
 m=initial('joint',c['seed']);m.load_state_dict(torch.load(OUT/f"control_{c['seed']}_{c['task']}.pt",weights_only=False)['state']);assert scores(m,c['seed'])==c['all_scores']
lines=['# 第六轮：顺序持续学习实测','','检索到的另一对话建议是顺序学习并测遗忘；未取得对第五轮成绩的新回复。本轮完成12条连续训练流（36个适应阶段）及9个独立适应对照。每阶段60轮、三种子。A为第五轮预训练任务，此后B→C→D无回放。新增任务身份明确可见。','','| 设置 | 初始A | 刚学完B/C/D平均 | 最终四任务平均 | 旧任务遗忘 |','|---|---|---|---|---|']
def fmt(a):return f'{np.mean(a)*100:.2f} ± {np.std(a,ddof=1)*100:.2f}%'
for kind in ['frozen','slow','joint','gru']:
 rr=[r for r in rows if r['kind']==kind];lines.append('| '+kind+' | '+' | '.join([fmt([r['matrix'][0][0] for r in rr]),fmt([np.mean(r['acquisition'][1:]) for r in rr]),fmt([r['final_mean'] for r in rr]),fmt([r['forgetting_mean_old'] for r in rr])])+' |')
lines+=['','遗忘按A/B/C已学阶段最好测试分减最终分，再平均；单位为百分点（表中百分号仅表示数值尺度）。小遗忘必须结合新任务学会程度判断。','','## 平均任务×阶段矩阵','']
for kind in ['frozen','slow','joint','gru']:
 arr=np.mean([r['matrix'] for r in rows if r['kind']==kind],axis=0)*100
 lines += [f'### {kind}','','| 阶段结束 | A | B | C | D |','|---|---|---|---|---|']
 for i,row in enumerate(arr):lines.append('| '+['预训练A','学完B','学完C','学完D'][i]+' | '+' | '.join(f'{x:.2f}' for x in row)+' |')
 lines.append('')
lines+=['## 独立适应控制','','每个任务从相同预训练Flow起点独立正常更新W，未先学习中间任务。']
for t in [1,2,3]:lines.append(f'- {"ABCD"[t]}：'+fmt([c['score'] for c in controls if c['task']==t]))
lines+=['','## 边界与复现','','Flow三个学习率分支共享预训练权重，额外任务输入权重零初始化，已验证扩展前后初始前向一致。GRU预训练能力不同，不能据此做完全公平的架构优劣排名。任务是相互关联的合成路由规则，身份明确可见，只有一个顺序。冻结W时控制器与读出仍可训练。本轮没有回放、参数保护、局部塑性或任务专属读出。','','所有阶段固定训练60轮，最后状态继续学习，验证只记录而不选模型；旧测试集不参与训练。保存逐轮曲线，可查达90%验证分的适应速度（未达到记为未达到）。只有3个种子，全部结果保留。','','验证：45个阶段/控制检查点重载，180个任务测试分逐项相同；前向扩展一致、规则前身份不可见、固定张量检查通过。运行 python run.py，然后 python finish.py。']
(ROOT/'REPORT.md').write_text('\n'.join(lines));(ROOT/'verification.txt').write_text('PASS: 45 checkpoints, 180 task scores, zero-extension equivalence, no early task identity, training frozen-tensor assertions.\n')
print('\n'.join(lines))
# Explicit acquisition speed and signed retention change.
extra=[]
for r in rows:
 extra.append({'kind':r['kind'],'seed':r['seed'],'epochs_to_90pct':{t:next((i+1 for i,h in enumerate(hh) if h['validation']>=.9),None) for t,hh in r['histories'].items()},'signed_forgetting':[r['matrix'][t][t]-r['matrix'][-1][t] for t in range(3)]})
(OUT/'adaptation_diagnostics.json').write_text(json.dumps(extra,indent=2))

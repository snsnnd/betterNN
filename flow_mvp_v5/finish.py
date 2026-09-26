import json,copy
from pathlib import Path
import numpy as np
import torch
from run import build,data,evaluate,OUT
rows=json.loads((OUT/'metrics.json').read_text());assert len(rows)==36
for r in rows:
    m=build(r['kind'],r['window']);m.load_state_dict(torch.load(OUT/(r['name']+'.pt'),weights_only=False)['state']);m.eval()
    for i,(cond,steps) in enumerate([('id',20),('long',40)]):
        got=evaluate(m,data(r['seed']*1000+10+i,2048,steps));assert got==r[cond]
    x,meta=data(97,8)[:2];m.train();a=m(x,meta);m.eval();b=m(x,meta);assert torch.allclose(a,b,atol=1e-6)
lines=['# 第五轮：梯度窗口实测报告','','基于另一对话的新意见，完成36次配对训练。20步序列，规则第4步起可见；初两步写入A/B，最终输出。完整/10步/5步反传，三种子。以下为两个输出同时正确率，均值±样本标准差。','','| 模型 | 反传窗口 | 常规20步 | 测试40步 |','|---|---|---|---|']
for kind in ['frozen','slow','joint','gru']:
    for w in [0,10,5]:
        rr=[r for r in rows if r['kind']==kind and r['window']==w]
        vals=[]
        for c in ['id','long']:
            a=np.array([r[c]['pair_accuracy'] for r in rr])*100;vals.append(f'{a.mean():.2f} ± {a.std(ddof=1):.2f}%')
        lines.append(f'| {kind} | {w or "完整"} | '+ ' | '.join(vals)+' |')
lines+=['','## 解释范围','','完整反传初两步输入梯度均非零，截断后为零；训练前向与不截断前向数值一致。说明截断切断的是梯度路径，不是清除记忆状态。冻结W仍允许损失通过W传回控制器，不能理解为无法反向传播。','','截断采用仅末端损失，没有中间辅助任务或局部塑性。窗口短于事件—答案间隔后，早期写入缺少直接梯度监督；共享控制器仍可从后期时间步更新。本结果不能单独代表所有截断BPTT、Hebbian或持续学习算法。','','数据量1024/512/2048、60轮；各配对组数据、种子、批次顺序相同，仅ID验证选模型。参数与计算量未匹配。只有20步与40步，未完成建议的200步基准，也未开始A→B→C遗忘实验。不可直接对比第四轮不同预算的分数。','','## 验证','','36个检查点重载，72组测试指标逐项一致；前向截断一致性通过，训练时固定B/mask/W检查通过。原始逐种子成绩、验证曲线、配置和模型均保留。','','运行：`python run.py`，随后 `python finish.py`。需要PyTorch和NumPy。']
(OUT.parent/'REPORT.md').write_text('\n'.join(lines));print('\n'.join(lines));(OUT.parent/'verification.txt').write_text('PASS 36 checkpoints, 72 metric sets, numerical forward equivalence; gradient-boundary and frozen-parameter assertions in run.py.\n')

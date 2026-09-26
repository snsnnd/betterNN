"""读取完整三种子结果生成比较图，不重新训练。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import CONDITIONS
p=Path('results');summ=json.loads((p/'summary.json').read_text());summ.update(json.loads(Path('results_slowleak/summary.json').read_text()))
order=['readout_only','clock_flow','marker_upper','marker_flow','marker_slowleak','marker_memory','slow_backbone','controller_only','gru_small','gru64']
labels=['Readout only','Clock controller','Marker upper only','Marker flow (a=0.2)','Marker flow (a=0.1)','Learned memory gate','Slow backbone','Controllers only','GRU (8 units)','GRU (64 units)']
conds=list(CONDITIONS);data=np.array([[summ[v][c]['mean']*100 for c in conds] for v in order]);sd=np.array([[summ[v][c]['std']*100 for c in conds] for v in order])
fig,ax=plt.subplots(figsize=(11,7.4),layout='constrained');im=ax.imshow(data,cmap='YlGnBu',vmin=50,vmax=100,aspect='auto')
ax.set_yticks(range(len(order)),labels);ax.set_xticks(range(5),['ID / 12 steps','Shifted positions','Random positions','Delay / 24 steps','3x distractor noise'])
plt.setp(ax.get_xticklabels(),rotation=15,ha='right')
for i in range(len(order)):
 for j in range(5):ax.text(j,i,f'{data[i,j]:.1f}\n±{sd[i,j]:.1f}',ha='center',va='center',color='white' if data[i,j]>82 else '#152331',fontsize=10)
ax.set_title('Randomly marked information: accuracy across five conditions\nMean ± sample SD (percentage points), three seeds',loc='left',pad=15)
fig.colorbar(im,ax=ax,label='Accuracy (%)',shrink=.75);fig.savefig(p/'comparison.png',dpi=180);plt.close(fig)
# 单独显示固定0.1、0.2与可学习保持门，突出对照而非择优展示。
keys=['marker_flow','marker_slowleak','marker_memory'];fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained');pos=np.arange(5)
for i,k in enumerate(keys):
 vals=[100*summ[k][c]['mean'] for c in conds];err=[100*summ[k][c]['std'] for c in conds]
 ax.bar(pos+(i-1)*.25,vals,.25,yerr=err,capsize=3,label=['Fixed update 0.2','Fixed update 0.1','Learned update gate'][i],color=['#7a8cae','#c7a166','#138d87'][i])
ax.set_xticks(pos,['ID','Shifted','Random','Long delay','Strong noise']);ax.set_ylim(0,105);ax.set_ylabel('Accuracy (%)');ax.legend(loc='lower right');ax.set_title('Separating slow updates from adaptive state retention',loc='left');ax.spines[['right','top']].set_visible(False)
fig.savefig(p/'memory_comparison.png',dpi=180)

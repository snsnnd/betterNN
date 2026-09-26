"""用实际结果绘图，不重新训练。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('results'); rows=json.loads((p/'summary.json').read_text())
names=list(rows)
labels=['Readout only','Controllers only','Upper + readout','Lower + readout','Both + readout','Both + slow W','Vanilla RNN','Small MLP']
fig,ax=plt.subplots(figsize=(10,5.7),layout='constrained')
y=np.arange(len(names)); vals=[rows[v]['accuracy_mean']*100 for v in names];err=[rows[v]['accuracy_std']*100 for v in names]
ax.barh(y,vals,xerr=err,color=['#a9b2bc','#e0a13b','#728ac3','#728ac3','#158a83','#158a83','#a9b2bc','#a9b2bc'],capsize=3)
ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,104);ax.axvline(50,c='gray',ls='--',lw=1)
for i,v in enumerate(vals): ax.text(v+err[i]+.6,i,f'{v:.1f}%',va='center',fontsize=9)
ax.set_xlabel('Held-out accuracy (%) — mean ± sample SD, 3 seeds')
ax.set_title('Fixed graph + learned modulation: a minimal test',loc='left',fontweight='bold')
ax.spines[['top','right']].set_visible(False)
fig.savefig(p/'accuracy.png',dpi=180);plt.close(fig)
traces=[np.load(f) for f in sorted(p.glob('trace_seed*.npz'))]
a=np.stack([t['selected_injection'] for t in traces]);fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
for i,t in enumerate(a):ax.plot(range(1,9),t,alpha=.35,label=f'Seed {list([11,22,33])[i]}')
ax.plot(range(1,9),a.mean(0),c='#158a83',lw=3,label='Mean')
ax.axvspan(4.5,8,color='#eea6a6',alpha=.2,label='Distractor interval')
ax.set(xlabel='Time step',ylabel='Injection gate (0–1)',ylim=(0,1),title='Does the upper controller suppress late distractors?');ax.legend()
fig.savefig(p/'injection.png',dpi=180)

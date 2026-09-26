import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import OUT
rows=json.loads((OUT/'metrics.json').read_text())
fig,axs=plt.subplots(1,2,figsize=(10,4),sharey=True)
for ax,cond in zip(axs,['id','long']):
    for kind in ['frozen','slow','joint','gru']:
        vals=[[r[cond]['pair_accuracy']*100 for r in rows if r['kind']==kind and r['window']==w] for w in [0,10,5]]
        ax.errorbar(range(3),np.mean(vals,axis=1),yerr=np.std(vals,axis=1,ddof=1),marker='o',capsize=3,label=kind)
    ax.set_xticks(range(3),['Full','10','5']);ax.set_xlabel('Backward window');ax.set_title('20-step test' if cond=='id' else '40-step test');ax.grid(alpha=.2);ax.set_ylim(0,105)
axs[0].set_ylabel('Both outputs correct (%)');axs[1].legend();fig.suptitle('Delayed routing: 3 seeds, mean ± sample SD');fig.tight_layout();fig.savefig(OUT/'bptt_comparison.png',dpi=160)

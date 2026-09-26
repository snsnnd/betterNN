import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
root=Path(__file__).parent;s=json.loads((root/'summary.json').read_text())
fig,axes=plt.subplots(2,3,figsize=(13,7))
items=[('final_mean','Final mean accuracy (%)',100),('plasticity','Acquisition accuracy (%)',100),('forgetting','Forgetting (percentage points)',100),('final_selection_overlap','Final selected-node Jaccard',1),('final_selectivity_used','Task selectivity (used nodes)',1),('final_route_similarity','Route cosine similarity',1)]
for ax,(key,title,scale) in zip(axes.flat,items):
 for mode in ['dense','topk']:
  rr=[r for r in s if r['mode']==mode];xx=[r['n'] for r in rr]
  ax.errorbar(xx,[r[key]['mean']*scale for r in rr],yerr=[r[key]['std']*scale for r in rr],capsize=3,marker='o',label=mode)
  if key=='final_selection_overlap' and mode=='topk':ax.plot(xx,[r['initial_selection_overlap']['mean'] for r in rr],':',color='gray',label='topk before learning')
 ax.set_xscale('log',base=2);ax.set_xticks([64,128,256,512,1024],['64','128','256','512','1024']);ax.set_title(title);ax.set_xlabel('Potential nodes');ax.grid(alpha=.2)
axes[0,0].legend();axes[1,0].legend();fig.suptitle('Capacity × task-conditioned node selection: 5 seeds, mean ± SD');fig.tight_layout();fig.savefig(root/'scaling.png',dpi=160)

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import OUT
rows=json.loads((OUT/'metrics.json').read_text());fig,axs=plt.subplots(1,4,figsize=(13,3.4),sharey=True)
for ax,kind in zip(axs,['frozen','slow','joint','gru']):
 arr=np.mean([r['matrix'] for r in rows if r['kind']==kind],axis=0)*100
 im=ax.imshow(arr,vmin=0,vmax=100,cmap='Blues')
 for i in range(4):
  for j in range(4):ax.text(j,i,f'{arr[i,j]:.1f}',ha='center',va='center',color='white' if arr[i,j]>65 else 'black')
 ax.set_xticks(range(4),list('ABCD'));ax.set_yticks(range(4),['After A','After B','After C','After D']);ax.set_title(kind);ax.set_xlabel('Evaluated task')
fig.suptitle('Sequential learning: mean accuracy over 3 seeds (%)');fig.tight_layout();fig.savefig(OUT/'retention_matrix.png',dpi=160)

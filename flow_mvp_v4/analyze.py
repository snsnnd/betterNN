import json,itertools
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments import build
from route_analysis import topology

def main():
    torch.set_num_threads(1);root=Path('results');allrows={st:json.loads((root/st/'metrics.json').read_text()) for st in ['topology','salience','delay','delay_block']};summary={}
    for st,rows in allrows.items():
        summary[st]={}
        keys=sorted(set((f"{r['seed']}_{r['mode']}" if st=='topology' else r['mode']) for r in rows))
        for key in keys:
            rr=[r for r in rows if (f"{r['seed']}_{r['mode']}" if st=='topology' else r['mode'])==key];numeric=[k for k,v in rr[0].items() if isinstance(v,(int,float)) and k not in ['seed','mask_seed']]
            summary[st][key]={k:{'mean':float(np.mean([r[k] for r in rr])),'std':float(np.std([r[k] for r in rr],ddof=1)) if len(rr)>1 else 0.} for k in numeric}
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    rr=allrows['topology'];index={(r['seed'],r['mask_seed'],r['mode']):r for r in rr};pairs=[]
    for ws in [101,202]:
        for ms in [11,22,33,44,55]:
            a=index[ws,ms,'dynamic'];b=index[ws,ms,'no_route'];job={'study':'topology','mode':'dynamic','seed':ws,'mask_seed':ms};model=build(job)
            r={'weight_seed':ws,'mask_seed':ms,**topology(model.W,model.mask)}
            r.update({f'delta_{k}_pp':100*(a[k]-b[k]) for k in ['id','shift','delay']})
            if ws==101:r.update({f'dynamic_minus_static_{k}_pp':100*(a[k]-index[ws,ms,'static'][k]) for k in ['id','shift','delay']})
            pairs.append(r)
    (root/'controlled_topology_pairs.json').write_text(json.dumps(pairs,indent=2))
    corr=[]
    for ws in [101,202]:
        rows=[r for r in pairs if r['weight_seed']==ws]
        corr.append({'weight_seed':ws,**{f:float(spearmanr([r[f] for r in rows],[r['delta_id_pp'] for r in rows]).statistic) for f in ['mean_reachable_path','spectral_radius','effective_rank','undirected_clustering']}})
    (root/'controlled_exploratory_correlations.json').write_text(json.dumps(corr,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,ws in zip(axes,[101,202]):
        rows=[r for r in pairs if r['weight_seed']==ws]
        for k,color in [('id','#108b82'),('shift','#a98443'),('delay','#5366b2')]:ax.plot([r['mask_seed'] for r in rows],[r[f'delta_{k}_pp'] for r in rows],marker='o',label=k,color=color)
        ax.axhline(0,color='gray',ls='--');ax.set(xlabel='Mask seed',ylabel='Full minus I+H (percentage points)',title=f'Fixed data and weight seed {ws}');ax.legend()
    fig.savefig(root/'controlled_topology.png',dpi=180);plt.close(fig)
    for st,cols,labels in [('salience',['id','shift','delay'],['ID','Shifted events','Long delay']),('delay',['id','early_rule','late_rule','long_hold'],['ID rule','Earlier rule','Later rule / 24','Long hold / 24'])]:
        order=['perfect','noisy','learned','none','gru'] if st=='salience' else ['no_route','dynamic','static','slow','gru']
        data=np.array([[summary[st][m][c]['mean']*100 for c in cols] for m in order]);sd=np.array([[summary[st][m][c]['std']*100 for c in cols] for m in order])
        fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained');im=ax.imshow(data,cmap='YlGnBu',vmin=0 if st=='delay' else 50,vmax=100,aspect='auto');ax.set_xticks(range(len(cols)),labels);ax.set_yticks(range(5),order)
        for i in range(5):
            for j in range(len(cols)):ax.text(j,i,f'{data[i,j]:.1f}\n±{sd[i,j]:.1f}',ha='center',va='center',color='white' if data[i,j]>(65 if st=='delay' else 82) else '#14242d')
        ax.set_title(('Learnable relevance from observable keys' if st=='salience' else 'Delayed rule: BOTH outputs must be correct')+'\nMean ± sample SD, 3 seeds',loc='left');fig.colorbar(im,ax=ax,label='Accuracy (%)',shrink=.8);fig.savefig(root/(st+'.png'),dpi=180);plt.close(fig)
    old=json.loads((root/'old_route_analysis.json').read_text());fig,ax=plt.subplots(figsize=(8,4),layout='constrained');pos=np.arange(5)
    for j,k in enumerate(['id_dynamic_accuracy','id_global_static_accuracy','id_task_static_accuracy']):ax.bar(pos+(j-1)*.25,[100*r[k] for r in old],.25,label=['Original dynamic','Global fixed gate','Task-fixed gate'][j])
    ax.set_xticks(pos,[r['seed'] for r in old]);ax.set(xlabel='Original seed',ylabel='ID accuracy (%)',ylim=(0,105),title='Inference interventions (gates estimated on training data)');ax.legend(loc='lower left');fig.savefig(root/'static_intervention.png',dpi=180)
    print('ANALYSIS COMPLETE')
if __name__=='__main__':main()

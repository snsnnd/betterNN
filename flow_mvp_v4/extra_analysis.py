"""独立新测试上的原延迟基线复评，以及显著性编码表。"""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments import *

def main():
    torch.set_num_threads(1);root=Path('results');old=json.loads((root/'delay/metrics.json').read_text());fresh=[]
    for r in old:
        job={k:r[k] for k in ['study','mode','seed']};model=build(job);model.load_state_dict(torch.load(root/'delay'/(r['run_id']+'.pt'),weights_only=True,map_location='cpu')['state_dict']);model.eval()
        _,_,tests=datasets({'study':'delay_block','mode':'block_dynamic','seed':r['seed']})
        fresh.append({'mode':r['mode'],'seed':r['seed'],**{k:score(model,d) for k,d in tests.items()},**diagnostics(model,job,tests['id'])})
    for r in json.loads((root/'delay_block/metrics.json').read_text()):fresh.append({k:v for k,v in r.items() if k not in ['run_id','seconds','trainable','best_epoch','val_accuracy','study']})
    (root/'fresh_delay_comparison.json').write_text(json.dumps(fresh,indent=2))
    modes=['no_route','dynamic','static','slow','block_dynamic','block_static','gru'];conds=['id','early_rule','late_rule','long_hold'];summary={}
    for mode in modes:
        rr=[r for r in fresh if r['mode']==mode];summary[mode]={k:{'mean':float(np.mean([r[k] for r in rr])),'std':float(np.std([r[k] for r in rr],ddof=1))} for k in conds+['opposite_exact_accuracy','counterfactual_both_correct']}
    (root/'fresh_delay_summary.json').write_text(json.dumps(summary,indent=2))
    data=np.array([[summary[m][c]['mean']*100 for c in conds] for m in modes]);fig,ax=plt.subplots(figsize=(9,5),layout='constrained');im=ax.imshow(data,cmap='YlGnBu',vmin=0,vmax=100,aspect='auto');ax.set_yticks(range(7),modes);ax.set_xticks(range(4),conds)
    for i in range(7):
        for j in range(4):ax.text(j,i,f'{data[i,j]:.1f}',ha='center',va='center',color='white' if data[i,j]>65 else '#122431')
    ax.set_title('Exploratory branch: all models on the SAME fresh test sets',loc='left');fig.colorbar(im,ax=ax,label='Both outputs correct (%)');fig.savefig(root/'fresh_delay.png',dpi=180);plt.close(fig)
    maps=[];fig,axes=plt.subplots(1,3,figsize=(11,3.6),layout='constrained')
    for ax,seed in zip(axes,[11,22,33]):
        job={'study':'salience','mode':'learned','seed':seed};model=build(job);model.load_state_dict(torch.load(root/'salience'/f'salience_learned_{seed}.pt',weights_only=True,map_location='cpu')['state_dict']);model.eval()
        q=torch.eye(4).repeat_interleave(4,0);key=torch.eye(4).repeat(4,1)
        with torch.no_grad():table=model.salience(torch.cat([q,key],1)).sigmoid().reshape(4,4).numpy()
        # 每查询目标键对三个非目标键的秩；0也可能是反向相关性编码。
        perq=[float(np.mean((table[k,k]>np.delete(table[k],k)).astype(float)+.5*(table[k,k]==np.delete(table[k],k)))) for k in range(4)]
        maps.append({'seed':seed,'query_by_key_table':table.tolist(),'per_query_target_rank_auc':perq})
        im=ax.imshow(table,vmin=0,vmax=1,cmap='viridis');ax.set(xlabel='Event key',ylabel='Query',title=f'Learned salience, seed {seed}',xticks=range(4),yticks=range(4))
        for i in range(4):
            for j in range(4):ax.text(j,i,f'{table[i,j]:.2f}',ha='center',va='center',color='white' if table[i,j]<.55 else '#152031')
    fig.colorbar(im,ax=axes,label='Encoder output');fig.savefig(root/'salience_lookup.png',dpi=180);(root/'salience_lookup.json').write_text(json.dumps(maps,indent=2));print('EXTRA ANALYSIS COMPLETE')
if __name__=='__main__':main()

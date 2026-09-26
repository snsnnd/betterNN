"""汇总原始五种子结果；配对差值与生成分布的解析参考不参与选模。"""
import itertools,json,math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import dataset,corrupt

@torch.no_grad()
def oracle_reference(seed,mode):
    x,q,m,y=dataset(4096,seed*1000+10);obs=corrupt(m,mode,seed*1000+100)
    v=x.gather(2,q.argmax(1)[:,None,None].expand(-1,12,1)).squeeze(-1)
    pairs=list(itertools.combinations(range(6),2));templates=torch.zeros(15,12)
    candidate_labels=torch.stack([(v[:,i]+v[:,j])>0 for i,j in pairs],1).float()
    for k,(i,j) in enumerate(pairs):templates[k,i]=templates[k,j]=1
    if mode=='none':posterior=torch.full_like(candidate_labels,1/15)
    elif mode=='noisy':
        dist=(obs[:,None,:]!=templates[None,:,:]).sum(-1).float()
        posterior=torch.softmax(-dist*math.log(.85/.15),1)
    else:
        posterior=(obs[:,None,:]==templates[None,:,:]).all(-1).float()
    prob=(posterior*candidate_labels).sum(1)
    return {'seed':seed,'cue':mode,'conditional_optimal_expected_accuracy':torch.maximum(prob,1-prob).mean().item(),'oracle_empirical_accuracy':((prob>.5)==y.bool()).float().mean().item()}

def main():
    torch.set_num_threads(1);root=Path('results');s=json.loads((root/'summary.json').read_text());rows=json.loads((root/'metrics.json').read_text());cfg=json.loads((root/'config.json').read_text());seeds=cfg['seeds'];indexed={(r['variant'],r['seed']):r for r in rows}
    paired=[]
    for first,second in [('g111','g101'),('g101','g100'),('g111_slow','g111'),('g111_joint','g111_slow')]:
        for c in ['id12','shift12','random12','delay24','noise12']:
            d=[100*(indexed[first,seed][c]-indexed[second,seed][c]) for seed in seeds]
            paired.append({'first':first,'second':second,'condition':c,'difference_pp':d,'mean_pp':float(np.mean(d)),'std_pp':float(np.std(d,ddof=1))})
    (root/'paired_differences.json').write_text(json.dumps(paired,indent=2))
    refs=[oracle_reference(seed,mode) for seed in seeds for mode in ['clean','noisy','none']]
    (root/'cue_oracle_reference.json').write_text(json.dumps(refs,indent=2))
    codes=[f'g{i}{r}{h}' for i,r,h in itertools.product([0,1],repeat=3)];conds=['id12','shift12','random12','delay24','noise12']
    labels=['No gates','Hold','Route','Route + Hold','Input','Input + Hold','Input + Route','Input + Route + Hold']
    data=np.array([[100*s[v][c]['mean'] for c in conds] for v in codes]);sd=np.array([[100*s[v][c]['std'] for c in conds] for v in codes])
    fig,ax=plt.subplots(figsize=(10,6),layout='constrained');im=ax.imshow(data,cmap='YlGnBu',vmin=50,vmax=100,aspect='auto');ax.set_yticks(range(8),labels);ax.set_xticks(range(5),['ID','Shifted positions','Random positions','Long delay','Strong noise'])
    for i in range(8):
        for j in range(5):ax.text(j,i,f'{data[i,j]:.1f}\n±{sd[i,j]:.1f}',ha='center',va='center',color='white' if data[i,j]>82 else '#102432',fontsize=10)
    ax.set_title('Retrained gate ablations: mean ± sample SD, five seeds',loc='left',pad=14);fig.colorbar(im,ax=ax,label='Accuracy (%)',shrink=.8);fig.savefig(root/'gate_ablation.png',dpi=180);plt.close(fig)
    # 在同一测试可靠性下比较清洁训练与适应训练。
    vc=['g101','g101_noisy','g101_none','gru','gru_noisy','gru_none'];cols=['clean_id','noisy_id','none_id'];data=np.array([[100*s[v][c]['mean'] for c in cols] for v in vc])
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained');im=ax.imshow(data,cmap='YlGnBu',vmin=50,vmax=100,aspect='auto');ax.set_yticks(range(6),['I+H / clean train','I+H / noisy train','I+H / no-cue train','GRU / clean train','GRU / noisy train','GRU / no-cue train']);ax.set_xticks(range(3),['Clean test cue','15% flip test cue','No test cue'])
    for i in range(6):
        for j in range(3):ax.text(j,i,f'{data[i,j]:.1f}',ha='center',va='center',color='white' if data[i,j]>82 else '#102432')
    ax.set_title('Training adaptation versus damaged-cue intervention',loc='left');fig.colorbar(im,ax=ax,label='Accuracy (%)',shrink=.8);fig.savefig(root/'cue_reliability.png',dpi=180);plt.close(fig)
    history=json.loads((root/'history.json').read_text());fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,code in zip(axes,['g101','g111']):
        h=history[f'{code}_22']['gate_history']
        for c in range(4):ax.plot([r['epoch'] for r in h],[r['channels'][c]['valid_injection'] for r in h],marker='.',label=f'Channel {c}')
        ax.set(xlabel='Epoch',ylabel='Valid target injection',ylim=(0,1.03),title=f'{code}, seed 22 (training subset)');ax.legend()
    fig.savefig(root/'gate_history_seed22.png',dpi=180)
    print('Saved paired differences, oracle references, and three figures.')
if __name__=='__main__':main()

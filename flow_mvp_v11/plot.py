"""第十一轮：replay 比例曲线与 Pareto 图。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
data = json.loads((ROOT / 'summary.json').read_text())
pooled = data['pooled']
per_order = data['per_order']
labels = sorted({e['label'] for e in pooled}, key=lambda x: float(x))
colors = {'all': '#8a8f98', 'freeze_W': '#c27e68', 'replay': '#169b90'}
x = np.arange(len(labels))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout='constrained')
for ax, key, title in [(axes[0], 'final_mean', 'Final mean accuracy (%)'),
                       (axes[1], 'forgetting', 'Forgetting (percentage points)')]:
    for arm in ['all', 'freeze_W']:
        rr = [next(e for e in pooled if e['arm'] == arm and e['label'] == lab) for lab in labels]
        ax.errorbar(x, [e[key]['mean'] * 100 for e in rr], yerr=[e[key]['std'] * 100 for e in rr],
                    marker='o', capsize=3, color=colors[arm], label=arm)
        for order_key, ee in per_order.items():
            r2 = [next(e for e in ee if e['arm'] == arm and e['label'] == lab) for lab in labels]
            ax.plot(x, [e[key]['mean'] * 100 for e in r2], color=colors[arm], alpha=.25, lw=.8)
    ax.set_xticks(x, [f'{lab}%' for lab in labels])
    ax.set_xlabel('replay share of each step')
    ax.set_title(title)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)

ax = axes[2]
for arm in ['all', 'freeze_W']:
    for e in [e for e in pooled if e['arm'] == arm]:
        ax.scatter(e['acquisition']['mean'] * 100, e['final_mean']['mean'] * 100,
                   color=colors[arm], s=45, zorder=3)
        ax.annotate(f"r={e['label']}", (e['acquisition']['mean'] * 100, e['final_mean']['mean'] * 100),
                    fontsize=7, xytext=(3, 3), textcoords='offset points')
ax.set_xlabel('acquisition accuracy (%)')
ax.set_ylabel('final mean accuracy (%)')
ax.set_title('Pareto: plasticity vs retention')
ax.grid(alpha=.25)

fig.suptitle('v11 replay ratio × {all, freeze_W} on A\'B\'C\'D\' (3 orders, 5 seeds)')
fig.savefig(ROOT / 'replay_matrix.png', dpi=160)
print('wrote replay_matrix.png')

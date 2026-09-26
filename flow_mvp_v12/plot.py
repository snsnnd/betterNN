"""第十二轮：架构 replay 曲线与所需比例图。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
summary = json.loads((ROOT / 'summary.json').read_text())
pooled = summary['pooled']
per_order = summary['per_order']
required = summary['required_ratio']
labels = sorted({e['label'] for e in pooled['flow']}, key=lambda x: float(x))
colors = {'flow': '#169b90', 'gru': '#c27e68', 'rnn': '#5366b2'}
x = np.arange(len(labels))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout='constrained')
for ax, key, title in [(axes[0], 'forgetting', 'Forgetting (percentage points)'),
                       (axes[1], 'final_mean', 'Final mean accuracy (%)')]:
    for arch in pooled:
        rr = [next(e for e in pooled[arch] if e['label'] == lab) for lab in labels]
        ax.errorbar(x, [e[key]['mean'] * 100 for e in rr], yerr=[e[key]['std'] * 100 for e in rr],
                    marker='o', capsize=3, color=colors[arch], label=arch)
        for order_key, ee in per_order.items():
            r2 = [next(e for e in ee if e['arch'] == arch and e['label'] == lab) for lab in labels]
            ax.plot(x, [e[key]['mean'] * 100 for e in r2], color=colors[arch], alpha=.25, lw=.8)
    ax.set_xticks(x, [f'{lab}%' for lab in labels])
    ax.set_xlabel('replay share of each step')
    ax.set_title(title)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)

ax = axes[2]
th = ['0.05', '0.03', '0.01']
w = .25
for i, arch in enumerate(pooled):
    vals = []
    for t in th:
        v = required[arch][t]
        vals.append(25 if v is None else v * 100)
    ax.bar(np.arange(len(th)) + (i - 1) * w, vals, width=w, color=colors[arch], label=arch)
    for j, t in enumerate(th):
        v = required[arch][t]
        ax.text(j + (i - 1) * w, vals[j] + .5, 'not reached' if v is None else f'{vals[j]:.1f}', ha='center', fontsize=7)
ax.set_xticks(range(len(th)), ['forgetting ≤5pp', '≤3pp', '≤1pp'])
ax.set_ylabel('required replay share (%)')
ax.set_title('Replay needed for a forgetting target')
ax.set_ylim(0, 28)
ax.grid(alpha=.25, axis='y')
ax.legend(fontsize=8)

fig.suptitle('v12 equal-budget replay: Flow (66k) vs GRU (67k) vs RNN (67k), A\'B\'C\'D\', 3 orders, 5 seeds')
fig.savefig(ROOT / 'architecture_replay.png', dpi=160)
print('wrote architecture_replay.png')

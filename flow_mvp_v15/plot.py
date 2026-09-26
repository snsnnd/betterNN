"""第十五轮：Flow-v2 基线图（与旧 Flow 对照）。"""
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
old = summary['old_flow_v12']
labels = [e['label'] for e in pooled]
x = np.arange(len(labels))

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), layout='constrained')
for ax, key, title in [(axes[0], 'forgetting', 'Forgetting (percentage points)'),
                       (axes[1], 'final_mean', 'Final mean accuracy (%)')]:
    ax.errorbar(x, [e[key]['mean'] * 100 for e in pooled], yerr=[e[key]['std'] * 100 for e in pooled],
                marker='o', capsize=3, color='#169b90', label='Flow-v2')
    for order_key, ee in per_order.items():
        ax.plot(x, [e[key]['mean'] * 100 for e in ee], color='#169b90', alpha=.25, lw=.8)
    if old:
        oy = [old[lab][key]['mean'] * 100 for lab in labels if lab in old]
        ox = [i for i, lab in enumerate(labels) if lab in old]
        ax.plot(ox, oy, marker='s', ls='--', color='#8a8f98', label='old Flow (v12)')
    ax.set_xticks(x, [f'{lab}%' for lab in labels])
    ax.set_xlabel('replay share of each step')
    ax.set_title(title)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)

fig.suptitle('v15 Flow-v2 baseline: W+Route+Hold+Readout (Write removed), 3 orders, 5 seeds')
fig.savefig(ROOT / 'flowv2_baseline.png', dpi=160)
print('wrote flowv2_baseline.png')

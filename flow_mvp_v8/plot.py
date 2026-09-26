"""从第八轮汇总生成归因图。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
summary = json.loads((ROOT / 'summary.json').read_text())
rows = json.loads((ROOT / 'all_metrics.json').read_text())
modes = [s['mode'] for s in summary]
colors = {'dense': '#8a8f98', 'fixed_topk': '#c27e68', 'learned_topk': '#169b90'}


def series(key, seed):
    return [next(x[key] for x in rows if x['mode'] == m and x['seed'] == seed) * 100 for m in modes]


fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
for ax, key, title in [(axes[0, 0], 'final_mean', 'Final mean accuracy (%)'),
                       (axes[0, 1], 'forgetting', 'Forgetting (percentage points)'),
                       (axes[1, 0], 'plasticity', 'Acquisition accuracy (%)')]:
    for seed in [11, 22, 33, 44, 55]:
        ax.plot(range(len(modes)), series(key, seed), marker='o', color='#9aa0a6', alpha=.35, lw=1)
    for mode in modes:
        vals = [r[key] * 100 for r in rows if r['mode'] == mode]
        ax.scatter([modes.index(mode)] * len(vals), vals, color=colors[mode], zorder=3, label=mode)
    ax.set_xticks(range(len(modes)), modes)
    ax.set_title(title)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)

ax = axes[1, 1]
x = np.arange(len(modes))
init = [next(s['initial_selection_overlap']['mean'] for s in summary if s['mode'] == m) for m in modes]
final = [next(s['final_selection_overlap']['mean'] for s in summary if s['mode'] == m) for m in modes]
iou = [next(s['support_iou_init_final']['mean'] for s in summary if s['mode'] == m) for m in modes]
ax.bar(x - .15, init, width=.3, label='selection overlap initial', color='#b8c4cc')
ax.bar(x + .15, final, width=.3, label='selection overlap final', color='#169b90')
ax.plot(x, iou, 'k^', label='support IoU init→final')
ax.set_xticks(x, modes)
ax.set_ylim(0, 1.05)
ax.set_title('Task overlap and selector change')
ax.legend(fontsize=8)
ax.grid(alpha=.25, axis='y')

fig.suptitle('Top-K attribution: dense vs fixed random vs learned selection (5 seeds)')
fig.savefig(ROOT / 'attribution.png', dpi=160)
print('wrote attribution.png')

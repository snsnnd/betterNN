"""第十三轮：机制图（梯度冲突、组件移植、缓冲大小）。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
summary = json.loads((ROOT / 'summary.json').read_text())
ctable = summary['conflict']['truncated']
t0 = summary['transplant']['0']
t12 = summary['transplant']['12.5']
buffer = summary['buffer']

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), layout='constrained')
groups = list(ctable)
x = np.arange(len(groups))
for i, ratio in enumerate([0, .125]):
    means = [np.mean([ctable[g][f'{ratio}_{s}']['cos_mean'] for s in [1, 2, 3]]) for g in groups]
    stds = [np.std([ctable[g][f'{ratio}_{s}']['cos_mean'] for s in [1, 2, 3]]) for g in groups]
    axes[0].bar(x + (i - .5) * .35, means, width=.35, yerr=stds, capsize=3,
                label=f"r={0 if ratio == 0 else 12.5}%", color=['#8a8f98', '#169b90'][i])
axes[0].axhline(0, color='gray', lw=.8)
axes[0].set_xticks(x, groups, rotation=20)
axes[0].set_ylabel('gradient cosine (newest vs old)')
axes[0].set_title('Task-gradient conflict by parameter group')
axes[0].legend(fontsize=8)
axes[0].grid(alpha=.25, axis='y')
axes[0].annotate('write: 0 under truncation\n(full BPTT nonzero)', xy=(1, 0.05), fontsize=7, ha='center')

labels_order = [(), ('W',), ('ctrl',), ('readout',), ('W', 'ctrl'), ('W', 'readout'), ('ctrl', 'readout'), ('W', 'ctrl', 'readout')]
names = ['none', '+W', '+ctrl', '+readout', '+W+ctrl', '+W+readout', '+ctrl+readout', 'all old']
mat0 = np.array([next(r['mean'] for r in t0 if tuple(r['combo']) == c) for c in labels_order])
mat12 = np.array([next(r['mean'] for r in t12 if tuple(r['combo']) == c) for c in labels_order])
im = axes[1].imshow(np.concatenate([mat0, mat12], axis=1), cmap='YlGnBu', vmin=45, vmax=100, aspect='auto')
axes[1].axvline(3.5, color='k', lw=1.2)
axes[1].set_xticks(range(8), ['A', 'B', 'C', 'D'] * 2)
axes[1].set_yticks(range(len(names)), names)
for i in range(8):
    for j in range(8):
        val = np.concatenate([mat0, mat12], axis=1)[i, j]
        axes[1].text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=6,
                     color='white' if val > 82 else '#182a32')
axes[1].set_title('Transplant (left: r=0%, right: r=12.5%)')

ax = axes[2]
if buffer:
    ms = [r['m'] for r in buffer]
    ax.errorbar(ms, [r['forgetting'][0] * 100 for r in buffer], yerr=[r['forgetting'][1] * 100 for r in buffer],
                marker='o', capsize=3, color='#c27e68', label='forgetting')
    ax.set_xscale('log', base=2)
    ax.set_xticks(ms, [str(m) for m in ms])
    ax.set_xlabel('replay buffer samples per task (r=12.5%)')
    ax.set_ylabel('forgetting (percentage points)')
    ax2 = ax.twinx()
    ax2.plot(ms, [r['final'][0] * 100 for r in buffer], marker='s', color='#169b90', label='final mean')
    ax2.set_ylabel('final mean accuracy (%)', color='#169b90')
    ax2.tick_params(axis='y', labelcolor='#169b90')
    ax.set_title('Buffer size scan (M=1..32)')
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, loc='upper right')

fig.suptitle('v13 mechanism: gradient conflict, component transplant, buffer size')
fig.savefig(ROOT / 'mechanism.png', dpi=160)
print('wrote mechanism.png')

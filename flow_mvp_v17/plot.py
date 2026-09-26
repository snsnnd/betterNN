"""第十七轮：可并行动力学对比图。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
table = json.loads((ROOT / 'summary.json').read_text())['table']
models = ['flowv2', 'linear', 'preroute', 'iter']
labels = ['A\n(flowv2)', 'B\n(linear)', 'C\n(preroute ✓)', 'D\n(iter ✓)']
colors = {'0': '#8a8f98', '12.5': '#169b90'}
x = np.arange(len(models))

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), layout='constrained')
for i, lab in enumerate(['0', '12.5']):
    vals = [next(e for e in table if e['model'] == m and e['label'] == lab) for m in models]
    axes[0].bar(x + (i - .5) * .38, [v['forgetting']['mean'] * 100 for v in vals], width=.38,
                yerr=[v['forgetting']['std'] * 100 for v in vals], capsize=3, color=colors[lab], label=f'replay {lab}%')
    axes[1].bar(x + (i - .5) * .38, [v['final40']['mean'] * 100 for v in vals], width=.38,
                yerr=[v['final40']['std'] * 100 for v in vals], capsize=3, color=colors[lab], label=f'replay {lab}%')
axes[0].set_title('CL forgetting (20-step)')
axes[1].set_title('Length extrapolation: final 40-step accuracy (%)')
for ax in axes[:2]:
    ax.set_xticks(x, labels)
    ax.grid(alpha=.25, axis='y')
    ax.legend(fontsize=8)

d0 = [next(e for e in table if e['model'] == m and e['label'] == '0')['drift_h']['mean'] for m in models]
d12 = [next(e for e in table if e['model'] == m and e['label'] == '12.5')['drift_h']['mean'] for m in models]
axes[2].bar(x - .19, d0, width=.38, color=colors['0'], label='replay 0%')
axes[2].bar(x + .19, d12, width=.38, color=colors['12.5'], label='replay 12.5%')
axes[2].set_xticks(x, labels)
axes[2].set_ylabel('mean |Δh| (final vs stage0)')
axes[2].set_title('State-trajectory drift on task A')
axes[2].grid(alpha=.25, axis='y')
axes[2].legend(fontsize=8)

fig.suptitle('v17 parallelizable dynamics: same architecture, four wirings (3 orders, 5 seeds)')
fig.savefig(ROOT / 'parallel_dynamics.png', dpi=160)
print('wrote parallel_dynamics.png')

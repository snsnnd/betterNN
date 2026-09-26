"""第十四轮：Write 归因图。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
summary = json.loads((ROOT / 'summary.json').read_text())['table']
modes = ['frozen', 'const1', 'const05', 'full_bptt', 'aux']
labels = ['frozen\n(random)', 'const1\n(p=1)', 'const05\n(p=0.5)', 'full_bptt\n(trained)', 'aux\n(local sup.)']
ratios = ['0', '12.5']
colors = {'0': '#8a8f98', '12.5': '#169b90'}

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), layout='constrained')
x = np.arange(len(modes))
for ax, key, title in [(axes[0], 'final_mean', 'Final mean accuracy (%)'),
                       (axes[1], 'forgetting', 'Forgetting (percentage points)')]:
    for i, label in enumerate(ratios):
        vals = [next(e for e in summary if e['write_mode'] == m and e['label'] == label) for m in modes]
        ax.bar(x + (i - .5) * .38, [v[key]['mean'] * 100 for v in vals], width=.38,
               yerr=[v[key]['std'] * 100 for v in vals], capsize=3, color=colors[label], label=f'replay {label}%')
    ax.set_xticks(x, labels)
    ax.set_title(title)
    ax.grid(alpha=.25, axis='y')
    ax.legend(fontsize=8)

ax = axes[2]
deltas = [next(e for e in summary if e['write_mode'] == m and e['label'] == '0')['write_delta']['mean'] for m in modes]
ax.bar(x, deltas, color=['#8a8f98', '#8a8f98', '#8a8f98', '#c27e68', '#a98443'])
ax.set_xticks(x, labels)
ax.set_ylabel('||ΔWrite|| after training')
ax.set_title('Was the write gate trained?')
ax.grid(alpha=.25, axis='y')

fig.suptitle('v14 Write causal attribution: frozen vs constant vs trained (r=0 and 12.5%)')
fig.savefig(ROOT / 'write_attribution.png', dpi=160)
print('wrote write_attribution.png')

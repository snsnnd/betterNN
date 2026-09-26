"""第九轮遗忘定位图：各组最终成绩、遗忘与 A 任务保留曲线。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
summary = json.loads((ROOT / 'summary.json').read_text())
rows = json.loads((ROOT / 'all_metrics.json').read_text())
arms = [s['arm'] for s in summary]
colors = {'all': '#8a8f98', 'freeze_W': '#c27e68', 'freeze_ctrl': '#b05050', 'freeze_readout': '#a98443',
          'ctrl_only': '#5366b2', 'readout_only': '#7a7fb0', 'replay': '#169b90'}

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout='constrained')
for ax, key, title in [(axes[0], 'final_mean', 'Final mean accuracy (%)'),
                       (axes[1], 'forgetting', 'Forgetting (percentage points)')]:
    vals = [next(s[key] for s in summary if s['arm'] == a) for a in arms]
    x = np.arange(len(arms))
    ax.bar(x, [v['mean'] * 100 for v in vals], yerr=[v['std'] * 100 for v in vals],
           color=[colors[a] for a in arms], capsize=3)
    ax.set_xticks(x, arms, rotation=25, ha='right')
    ax.set_title(title)
    ax.grid(alpha=.25, axis='y')

ax = axes[2]
for a in ['all', 'replay', 'freeze_W', 'ctrl_only']:
    m = np.mean([r['matrix'] for r in rows if r['arm'] == a], axis=0)
    ax.plot(range(4), [row[0] * 100 for row in m], marker='o', label=a, color=colors[a])
ax.set_xticks(range(4), ['after A', 'after B', 'after C', 'after D'])
ax.set_title('Task A accuracy across stages')
ax.set_ylabel('accuracy (%)')
ax.legend(fontsize=8)
ax.grid(alpha=.25)

fig.suptitle('v9 forgetting localization: frozen groups and fixed-budget replay (5 seeds)')
fig.savefig(ROOT / 'forgetting.png', dpi=160)
print('wrote forgetting.png')

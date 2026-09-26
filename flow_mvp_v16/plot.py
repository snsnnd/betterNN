"""第十六轮：Forgetting vs Memory Bytes。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
summary = json.loads((ROOT / 'summary.json').read_text())
table = summary['table']
methods = ['sample', 'route_hold', 'route', 'hold', 'hybrid']
colors = {'sample': '#8a8f98', 'route_hold': '#169b90', 'route': '#5366b2', 'hold': '#a98443', 'hybrid': '#c27e68'}
labels = {'sample': 'sample (x, y)', 'route_hold': 'route+hold anchors', 'route': 'route anchors',
          'hold': 'hold anchors', 'hybrid': 'hybrid (anchors + tiny samples)'}

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), layout='constrained')
for ax, key, title in [(axes[0], 'forgetting', 'Forgetting (percentage points)'),
                       (axes[1], 'final_mean', 'Final mean accuracy (%)')]:
    none = next(e for e in table if e['method'] == 'none')
    ax.axhline(none[key]['mean'] * 100, color='#cccccc', ls='--', lw=1)
    ax.text(1.05, none[key]['mean'] * 100 + (0.6 if key == 'forgetting' else 0.6), 'no replay', fontsize=7, color='#888888')
    for method in methods:
        pts = sorted([e for e in table if e['method'] == method], key=lambda e: e['bytes'])
        xs = [max(e['bytes'], 1) for e in pts]
        ys = [e[key]['mean'] * 100 for e in pts]
        es = [e[key]['std'] * 100 for e in pts]
        ax.errorbar(xs, ys, yerr=es, marker='o', capsize=3, color=colors[method], label=labels[method])
    ax.set_xscale('log')
    ax.set_xlabel('memory per task (bytes)')
    ax.set_title(title)
    ax.grid(alpha=.25)
    ax.legend(fontsize=7)

axes[0].axhline(3, color='k', ls=':', lw=.8)
axes[0].axhline(1, color='k', ls=':', lw=.8)

fig.suptitle('v16 policy replay: Route/Hold anchors vs raw samples (bandwidth fixed at 16/step)')
fig.savefig(ROOT / 'policy_replay.png', dpi=160)
print('wrote policy_replay.png')

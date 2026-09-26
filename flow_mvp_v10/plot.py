"""第十轮 benchmark 图：校准曲线与顺序 CL 保留矩阵。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
bench = json.loads((ROOT / 'benchmark.json').read_text())
curves = bench['curves']
cl = [json.loads(p.read_text()) for p in sorted((ROOT / 'results' / 'cl').glob('*.json'))]
labels = ["A'", "B'", "C'", "D'"]
colors = {'0': '#169b90', '1': '#c27e68', '2': '#5366b2', '3': '#a98443'}

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout='constrained')
ax = axes[0]
for key, c in curves.items():
    mode, task = key.rsplit('_', 1)
    ax.plot(range(1, len(c) + 1), [v * 100 for v in c], color=colors[task],
            ls='-' if mode == 'scratch' else '--', lw=1, label=f'{mode} {labels[int(task)]}')
ax.axhline(95, color='gray', ls=':', lw=.8)
ax.axhline(90, color='gray', ls=':', lw=.8)
for e in [bench['epochs_90'], bench['epochs_95']]:
    if e:
        ax.axvline(e, color='k', ls='--', lw=.8)
        ax.text(e, 52, f' E*={e}', fontsize=8)
ax.set_xlabel('epoch')
ax.set_ylabel('mean validation (%)')
ax.set_title("A'B'C'D' calibration: scratch (solid) vs adapt (dashed)")
ax.grid(alpha=.25)
ax.legend(fontsize=6, ncol=2)

ax = axes[1]
mat = np.mean([r['matrix'] for r in cl], axis=0) * 100
im = ax.imshow(mat, cmap='YlGnBu', vmin=40, vmax=100, aspect='auto')
ax.set_xticks(range(4), [f'after {n}' for n in labels])
ax.set_yticks(range(4), labels)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f'{mat[i, j]:.1f}', ha='center', va='center',
                color='white' if mat[i, j] > 82 else '#182a32')
ax.set_title('Sequential CL baseline (65 epochs/stage, 5 seeds)')
fig.colorbar(im, ax=ax, shrink=.9)
fig.suptitle("v10 benchmark: every task learnable alone (~99.8%)")
fig.savefig(ROOT / 'benchmark.png', dpi=160)
print('wrote benchmark.png')

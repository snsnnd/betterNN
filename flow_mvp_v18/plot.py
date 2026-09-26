"""第十八轮：有效深度 vs 精度 / 状态误差。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
s = json.loads((ROOT / 'summary.json').read_text())
table = s['table']
refs = s['refs']
Ls = [1, 2, 4, 5, 10]
cmap = {1: '#5366b2', 2: '#169b90', 4: '#c27e68', 5: '#a98443', 10: '#7a7fb0'}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout='constrained')
for ax, key, title in [(axes[0], 'acc20', '20-step accuracy (%)'), (axes[1], 'acc40', '40-step accuracy (%)')]:
    for init, marker, ls in [('preroute', 'o', '-'), ('zero', 's', '--')]:
        for L in Ls:
            pts = sorted([e for e in table if e['init'] == init and e['L'] == L], key=lambda e: e['depth'])
            ax.plot([e['depth'] for e in pts], [e[key] * 100 for e in pts], marker=marker, ls=ls,
                    color=cmap[L], alpha=.9, label=f'{init} L={L}')
    ax.axhline(refs['flow'][key] * 100, color='k', ls=':', lw=1)
    ax.axhline(refs['preroute_model'][key] * 100, color='gray', ls=':', lw=1)
    ax.axvline(20, color='k', lw=.6, alpha=.4)
    ax.text(20.3, 50, 'Flow depth 20', fontsize=7, rotation=90, color='#555555')
    ax.set_xlabel('effective sequential depth  (K+1)·L')
    ax.set_title(title)
    ax.grid(alpha=.25)
axes[0].legend(fontsize=6, ncol=2)

for init, marker, ls in [('preroute', 'o', '-'), ('zero', 's', '--')]:
    for L in Ls:
        pts = sorted([e for e in table if e['init'] == init and e['L'] == L], key=lambda e: e['depth'])
        vals = [max(e['Eh'], 1e-6) for e in pts]
        axes[2].plot([e['depth'] for e in pts], vals, marker=marker, ls=ls, color=cmap[L], alpha=.9)
axes[2].set_yscale('log')
axes[2].set_xlabel('effective sequential depth')
axes[2].set_ylabel('boundary state relative error $E_h$')
axes[2].set_title('State error vs depth')
axes[2].grid(alpha=.25)

fig.suptitle('v18 block-affine compressibility of Flow-v2 (r=12.5%, 5 seeds, 4 tasks): markers o=preroute init, s=zero init')
fig.savefig(ROOT / 'compressibility.png', dpi=160)
print('wrote compressibility.png')

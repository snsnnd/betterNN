"""第十九轮 Phase 0B 绘图：scan tree 秩增长 / 截断误差 / 求解收敛。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
S = json.loads((ROOT / 'results/summary.json').read_text())
LS = [1, 2, 4, 5, 10]
RKEYS = ['4', '8', '16', '32', '64']
INITS = ['pre', 'zero', 'noise0.05', 'noise0.2']

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
for L in LS:
    lv = S['tree']['40'][str(L)]['levels']
    x = list(range(1, len(lv) + 1))
    y = [v['rank_mean'] for v in lv]
    ax.plot(x, y, marker='o', label=f'L={L}')
ax.set_xlabel('scan tree level')
ax.set_ylabel('significant rank of correction')
ax.set_title('40-step: correction rank saturates (tol 1e-4)')
ax.axhline(256, color='k', ls=':', lw=0.8)
ax.legend(fontsize=8)

ax = axes[1]
for L in LS:
    t = S['tree']['40'][str(L)]['trunc']
    y = [t[r]['C_rel_err'] for r in RKEYS]
    ax.semilogy([int(r) for r in RKEYS], y, marker='o', label=f'L={L}')
ax.set_xscale('log', base=2)
ax.set_xlabel('recompression rank r')
ax.set_ylabel('relative error of correction')
ax.set_title('40-step: truncation error vs rank')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / 'operator_structure.png', dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for k, steps in enumerate([20, 40]):
    ax = axes[k]
    for iname in INITS:
        vals = [S['side'][str(steps)][f'{iname}/full'][f'Eh_glob_K{i}'] for i in range(4)]
        ax.semilogy(range(4), np.maximum(vals, 1e-10), marker='o', label=iname)
    ax.set_xlabel('relinearization passes K')
    ax.set_ylabel('Eh_glob')
    ax.set_title(f'{steps}-step solver convergence (mean over jobs)')
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / 'solver_convergence.png', dpi=150)
plt.close(fig)

bench = ROOT / 'results/solver_bench_zero.json'
if bench.exists():
    rows = json.loads(bench.read_text())['rows']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    names = ['serial', 'gs_jvp', 'structured', 'structured_seq']
    for ax, batch in [(axes[0], 1), (axes[1], 8)]:
        for name in names:
            xs, ys = [], []
            for r in rows:
                if r['batch'] == batch and r['K'] == 0 and name in r and 'latency_s' in r[name]:
                    xs.append(r['steps'])
                    ys.append(r[name]['latency_s'] * 1e3)
            if xs:
                order = np.argsort(xs)
                ax.loglog(np.array(xs)[order], np.array(ys)[order], marker='o', label=name)
        ax.set_xlabel('sequence length T')
        ax.set_ylabel('latency (ms)')
        ax.set_title(f'zero centers, K=0, batch={batch}')
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / 'solver_pareto.png', dpi=150)
    plt.close(fig)
    print('wrote operator_structure.png, solver_convergence.png, solver_pareto.png')
else:
    print('wrote operator_structure.png, solver_convergence.png')

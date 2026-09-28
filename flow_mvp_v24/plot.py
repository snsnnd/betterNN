"""第二十四轮绘图：acc vs T、retention vs T、lifetime、Phase B。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
KERNELS = ['single', 'burst3', 'burst5', 'decay-fast', 'decay-slow']
COL = {'single': '#1f77b4', 'burst3': '#2ca02c', 'burst5': '#d62728',
       'decay-fast': '#ff7f0e', 'decay-slow': '#9467bd'}
CHAIN_KERNELS = ['single', 'burst5', 'decay-slow']


def load(d):
    p = ROOT / d
    return [json.loads(f.read_text()) for f in sorted(p.glob('*.json'))] if p.exists() else []


rows = [r for r in load('results/delay') if r.get('phase') == 'delay']
chain = [r for r in load('results/chain') if r.get('phase') == 'chain']
Ts = sorted({r['T'] for r in rows})

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

# 1) acc vs T
ax = axes[0]
for k in KERNELS:
    xs, ys, es = [], [], []
    for T in Ts:
        v = [r['final'] for r in rows if r['kernel'] == k and r['T'] == T]
        if v:
            xs.append(T)
            ys.append(np.mean(v))
            es.append(np.std(v, ddof=1) / np.sqrt(len(v)))
    ax.errorbar(xs, ys, yerr=es, marker='o', capsize=3, color=COL[k], label=k)
ax.set_xscale('log', base=2)
ax.set_xticks(Ts, [str(t) for t in Ts])
ax.set_xlabel('T')
ax.set_ylabel('final accuracy')
ax.set_title('write kernel: accuracy vs T (4 tasks x 5 seeds)')
ax.legend(fontsize=8)
ax.set_ylim(0, 1.05)

# 2) retention at T/2
ax = axes[1]
for k in KERNELS:
    xs, ys = [], []
    for T in Ts:
        v = [r['retention']['ret_half'] for r in rows if r['kernel'] == k and r['T'] == T
             and np.isfinite(r['retention']['ret_half'])]
        if v:
            xs.append(T)
            ys.append(np.mean(v))
    ax.plot(xs, ys, marker='o', color=COL[k], label=k)
ax.set_xscale('log', base=2)
ax.set_xticks(Ts, [str(t) for t in Ts])
ax.set_xlabel('T')
ax.set_ylabel('||dh(T/2)|| / ||dh(5)||')
ax.set_title('early-input response retention at T/2')
ax.legend(fontsize=8)

# 3) decay slope
ax = axes[2]
for k in KERNELS:
    xs, ys = [], []
    for T in Ts:
        v = [r['retention']['decay_slope'] for r in rows if r['kernel'] == k and r['T'] == T
             and np.isfinite(r['retention']['decay_slope'])]
        if v:
            xs.append(T)
            ys.append(np.mean(v))
    ax.plot(xs, ys, marker='o', color=COL[k], label=k)
ax.axhline(0, color='k', lw=.5)
ax.set_xscale('log', base=2)
ax.set_xticks(Ts, [str(t) for t in Ts])
ax.set_xlabel('T')
ax.set_ylabel('log-decay slope of ||dh_t||')
ax.set_title('perturbation decay along the delay')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / 'write_dynamics.png', dpi=150)
plt.close(fig)

if chain:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, task in zip(axes, ('xor', 'xorsw')):
        xs = np.arange(len(CHAIN_KERNELS))
        final = [np.mean([r['final'] for r in chain if r['task'] == task and r['kernel'] == k])
                 for k in CHAIN_KERNELS]
        best = [np.mean([r['best_val'] for r in chain if r['task'] == task and r['kernel'] == k])
                for k in CHAIN_KERNELS]
        ax.bar(xs - .2, final, width=.4, label='final', alpha=.85)
        ax.bar(xs + .2, best, width=.4, label='best-val', alpha=.55)
        ax.set_xticks(xs, CHAIN_KERNELS)
        ax.set_ylim(0, 1.05)
        ax.set_title(f'Chain-select T=80 {task} (K=5 hybrid)')
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / 'chain_kernels.png', dpi=150)
    plt.close(fig)
    print('wrote write_dynamics.png, chain_kernels.png')
else:
    print('wrote write_dynamics.png')

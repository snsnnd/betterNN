"""第二十三轮绘图：accuracy vs T、K 学习曲线、y0/y1 分解、gap。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
DIRS = ['results/main', 'results/control', 'results/scale']
KS = ['5', '10', '20', 'full']
COL = {'5': '#d62728', '10': '#ff7f0e', '20': '#2ca02c', 'full': '#1f77b4'}


def load_rows():
    rows = []
    for d in DIRS:
        p = ROOT / d
        if p.exists():
            for f in sorted(p.glob('*.json')):
                rows.append(json.loads(f.read_text()))
    return rows


rows = load_rows()


def sel(variant, task, T, K):
    return [r for r in rows if r['variant'] == variant and r['task'] == task
            and r['T'] == T and r['K'] == K]


fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

# 1) accuracy vs T
ax = axes[0][0]
Ts = sorted({r['T'] for r in rows if r['variant'] == 'transient'})
for K in KS:
    ys, xs = [], []
    for T in Ts:
        rs = sel('transient', 'xor', T, K)
        if rs:
            xs.append(T)
            ys.append(np.mean([r['final'] for r in rs]))
    if ys:
        ax.plot(xs, ys, 'o-', color=COL[K], label=f'K={K}')
ax.set_xscale('log', base=2)
ax.set_xticks(Ts, [str(t) for t in Ts])
ax.set_xlabel('T')
ax.set_ylabel('test accuracy (xor)')
ax.set_title('transient xor: accuracy vs sequence length')
ax.legend(fontsize=8)
ax.set_ylim(0, 1.05)

# 2) learning curves at T=80
ax = axes[0][1]
for K in KS:
    rs = [r for r in sel('transient', 'xor', 80, K) if r.get('curve')]
    if not rs:
        continue
    epochs = [c['epoch'] for c in rs[0]['curve']]
    y = np.array([[c['acc'] for c in r['curve']] for r in rs])
    ax.plot(epochs, y.mean(0), color=COL[K], label=f'K={K}')
    ax.fill_between(epochs, y.min(0), y.max(0), color=COL[K], alpha=.15)
ax.set_xlabel('epoch')
ax.set_ylabel('val accuracy')
ax.set_title('T=80 transient xor: learning curves (mean, min-max band)')
ax.legend(fontsize=8)
ax.set_ylim(0, 1.05)

# 3) decomposition at T=80
ax = axes[1][0]
w = .18
for i, K in enumerate(KS):
    rs = sel('transient', 'xor', 80, K)
    if not rs:
        continue
    y0 = np.mean([r['y0'] for r in rs])
    y1 = np.mean([r['y1'] for r in rs])
    ax.bar(i - .2, y0, width=.4, color='#9467bd', alpha=.8)
    ax.bar(i + .2, y1, width=.4, color='#8c564b', alpha=.8)
ax.set_xticks(range(len(KS)), [f'K={k}' for k in KS])
ax.set_ylabel('accuracy')
ax.set_title('T=80 transient xor: y0 (selection) vs y1 (context readout)')
ax.plot([], [], color='#9467bd', label='y0 selection')
ax.plot([], [], color='#8c564b', label='y1 context')
ax.legend(fontsize=8)
ax.set_ylim(0, 1.05)

# 4) gap(Full - K5) by task and T + sustained control
ax = axes[1][1]
labels, vals = [], []
for task in ('xor', 'xorsw'):
    for T in Ts:
        a = {r['seed']: r['final'] for r in sel('transient', task, T, 'full')}
        b = {r['seed']: r['final'] for r in sel('transient', task, T, '5')}
        seeds = sorted(set(a) & set(b))
        if seeds:
            labels.append(f'{task}\nT{T}')
            vals.append(100 * np.mean([a[s] - b[s] for s in seeds]))
for task in ('xor', 'xorsw'):
    a = {r['seed']: r['final'] for r in sel('sustained', task, 80, 'full')}
    b = {r['seed']: r['final'] for r in sel('sustained', task, 80, '5')}
    seeds = sorted(set(a) & set(b))
    if seeds:
        labels.append(f'{task}\nT80 sus')
        vals.append(100 * np.mean([a[s] - b[s] for s in seeds]))
ax.bar(range(len(labels)), vals, alpha=.8)
ax.axhline(15, color='r', ls='--', lw=.8)
ax.set_xticks(range(len(labels)), labels, fontsize=7)
ax.set_ylabel('Full − K5 (pp)')
ax.set_title('credit-window gap (dashed = 0.15 criterion)')
fig.tight_layout()
fig.savefig(ROOT / 'credit_stress.png', dpi=150)
plt.close(fig)
print('wrote credit_stress.png')

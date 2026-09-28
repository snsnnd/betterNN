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
sched = [r for r in load('results/sched_chain') if r.get('phase') == 'sched']
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
    print('wrote chain_kernels.png')

if sched:
    ks = sorted({r['kernel'] for r in sched if r['T'] == 80})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    # 1) fixed vs sched (T=80)
    ax = axes[0]
    xs, labels = [], []
    w = .35
    for i, (task, k) in enumerate([(t, k) for t in ('xor', 'xorsw') for k in ks]):
        fm = np.mean([r['final'] for r in chain if r['task'] == task and r['kernel'] == k and r['T'] == 80])
        sm = np.mean([r['final'] for r in sched if r['task'] == task and r['kernel'] == k and r['T'] == 80])
        ax.bar(i - w / 2, fm, width=w, alpha=.85, color='#1f77b4', label='fixed' if i == 0 else None)
        ax.bar(i + w / 2, sm, width=w, alpha=.85, color='#d62728', label='sched' if i == 0 else None)
        labels.append(f'{task}\n{k}')
    ax.set_xticks(range(len(labels)), labels, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_title('T=80: fixed kernel vs learnable scheduler')
    ax.legend(fontsize=8)
    # 2) w profile
    ax = axes[1]
    events = ['A', 'c1', 'distractor', 'B', 'c2']
    xw = np.arange(len(events))
    for j, task in enumerate(('xor', 'xorsw')):
        for m, k in enumerate(ks):
            rs = [r for r in sched if r['task'] == task and r['kernel'] == k and r['T'] == 80]
            if not rs:
                continue
            ys = [np.mean([r['w_profile'][e] for r in rs]) for e in events]
            ax.bar(xw + (j * len(ks) + m - (len(ks) * 2 - 1) / 2) * .13, ys, width=.12,
                   label=f'{task}/{k}', alpha=.85)
    ax.axhline(1.0, color='k', lw=.5)
    ax.set_xticks(xw, events)
    ax.set_ylabel('mean w')
    ax.set_title('event-level write strength')
    ax.legend(fontsize=6, ncol=2)
    # 3) T=160
    ax = axes[2]
    tags, vals = [], []
    for tag, rs in [('single\nfixed', [r for r in chain if r['T'] == 160 and r['task'] == 'xor' and r['kernel'] == 'single']),
                    ('burst5\nfixed', [r for r in chain if r['T'] == 160 and r['task'] == 'xor' and r['kernel'] == 'burst5']),
                    ('burst5\n+sched', [r for r in sched if r['T'] == 160 and r['task'] == 'xor' and r['kernel'] == 'burst5'])]:
        if rs:
            tags.append(tag)
            vals.append(np.mean([r['final'] for r in rs]))
    ax.bar(range(len(tags)), vals, alpha=.85)
    for i, v in enumerate(vals):
        ax.text(i, v + .01, f'{v:.3f}', ha='center', fontsize=8)
    ax.set_xticks(range(len(tags)), tags, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title('T=160 xor: scheduler vs fixed')
    fig.tight_layout()
    fig.savefig(ROOT / 'scheduler.png', dpi=150)
    plt.close(fig)
    print('wrote scheduler.png')


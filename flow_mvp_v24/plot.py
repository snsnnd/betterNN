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

bank_chain = [r for r in load('results/bank_chain') if r.get('phase') == 'bank']
bank_delay = [r for r in load('results/bank_delay') if r.get('phase') == 'bank_delay']
if bank_chain or bank_delay:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    chain80 = [r for r in chain if r['T'] == 80]
    # 1) chain adaptive vs best fixed
    ax = axes[0][0]
    if bank_chain:
        labels, adaptive, bestf = [], [], []
        for task in ('xor', 'xorsw'):
            rs = [r for r in bank_chain if r['task'] == task]
            fixed = {k: np.mean([r['final'] for r in chain80 if r['task'] == task and r['kernel'] == k])
                     for k in CHAIN_KERNELS}
            bk = max(fixed, key=lambda k: fixed[k])
            labels.append(task + chr(10) + '(vs ' + bk + ')')
            adaptive.append(np.mean([r['final'] for r in rs]))
            bestf.append(fixed[bk])
        xs = np.arange(len(labels))
        ax.bar(xs - .2, adaptive, width=.4, label='adaptive', alpha=.85)
        ax.bar(xs + .2, bestf, width=.4, label='best fixed', alpha=.55)
        for i, (a, b) in enumerate(zip(adaptive, bestf)):
            ax.text(i, max(a, b) + .01, f'{a:.3f}/{b:.3f}', ha='center', fontsize=8)
        ax.set_xticks(xs, labels, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title('V24C chain T=80: adaptive vs best fixed')
        ax.legend(fontsize=8)
    # 2) alpha composition per event
    ax = axes[0][1]
    events = ['A', 'c1', 'distractor', 'B', 'c2']
    knames = ['single', 'burst3', 'burst5', 'dfast', 'dslow']
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd']
    width = .35
    for j, task in enumerate(('xor', 'xorsw')):
        rs = [r for r in bank_chain if r['task'] == task]
        if not rs:
            continue
        for m, e in enumerate(events):
            al = np.mean([r['profile'][e]['alpha'] for r in rs], axis=0)
            bottom = 0
            for k, c in zip(range(5), colors):
                ax.bar(m + (j - .5) * width, al[k], width=width, bottom=bottom, color=c,
                       alpha=.9 if j == 0 else .55)
                bottom += al[k]
    ax.set_xticks(range(len(events)), events)
    ax.set_ylabel('mean alpha')
    ax.set_title('kernel allocation (solid=xor, faded=xorsw)')
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in colors], labels=knames, fontsize=6)
    # 3) L_eff per event
    ax = axes[1][0]
    for j, task in enumerate(('xor', 'xorsw')):
        rs = [r for r in bank_chain if r['task'] == task]
        if not rs:
            continue
        ys = [np.mean([r['profile'][e]['L_eff'] for r in rs]) for e in events]
        ax.bar(np.arange(len(events)) + (j - .5) * .35, ys, width=.35, label=task, alpha=.85)
    ax.set_xticks(range(len(events)), events)
    ax.set_ylabel('L_eff = sum tau a_tau^2')
    ax.set_title('effective temporal length per event (chain)')
    ax.legend(fontsize=8)
    # 4) delay adaptive vs fixed
    ax = axes[1][1]
    Ts_d = sorted({r['T'] for r in bank_delay})
    if Ts_d:
        for tag, getter, c in [('adaptive', lambda T: [r['final'] for r in bank_delay if r['T'] == T], '#d62728'),
                               ('best fixed', lambda T: [np.mean([r['final'] for r in rows if r['T'] == T and r['kernel'] == k])
                                                          for k in []], None),
                               ('single', lambda T: [r['final'] for r in rows if r['T'] == T and r['kernel'] == 'single'], '#1f77b4')]:
            if tag == 'best fixed':
                ys = []
                for T in Ts_d:
                    vals = [np.mean([r['final'] for r in rows if r['T'] == T and r['kernel'] == k]) for k in ['single', 'burst3', 'burst5', 'decay-fast', 'decay-slow']]
                    ys.append(max(vals))
                ax.plot(Ts_d, ys, 's--', color='#2ca02c', label='best fixed')
                continue
            ys = [np.mean(getter(T)) for T in Ts_d]
            ax.plot(Ts_d, ys, 'o-', color=c, label=tag)
        ax.set_xscale('log', base=2)
        ax.set_xticks(Ts_d, [str(t) for t in Ts_d])
        ax.set_xlabel('T')
        ax.set_ylabel('final accuracy')
        ax.set_title('delay task: adaptive vs fixed')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(ROOT / 'bank.png', dpi=150)
    plt.close(fig)
    print('wrote bank.png')

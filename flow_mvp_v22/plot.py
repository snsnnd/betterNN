"""第二十二轮绘图：归因条形 / 主效应 / 信用窗口审计。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
ARMS = ['B', 'B+W', 'B+R', 'B+H', 'B+W+R', 'B+W+H', 'B+R+H', 'All']
GROUPS = ('W', 'route', 'hold', 'readout', 'B')


def cl_rows():
    rows = []
    for ph in ('A', 'B', 'C'):
        d = ROOT / f'results/{ph}'
        if not d.exists():
            continue
        for f in d.glob('*.json'):
            if f.name.startswith('metrics_') or f.name in ('config.json', 'summary.json', 'credit_audit.json'):
                continue
            r = json.loads(f.read_text())
            if isinstance(r, dict) and 'matrix' in r:
                rows.append(r)
    return rows


rows = cl_rows()
summ = json.loads((ROOT / 'results/summary.json').read_text()) if (ROOT / 'results/summary.json').exists() else None

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

# 1) forgetting by arm (r=12.5)
ax = axes[0]
xs, ys, es = [], [], []
for arm in ARMS:
    vals = [r['forgetting'] * 100 for r in rows if r['arm'] == arm and r['ratio'] == .125]
    if not vals:
        continue
    xs.append(arm)
    ys.append(np.mean(vals))
    es.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
ax.bar(xs, ys, yerr=es, capsize=3, alpha=.75, color=['#888', '#1f77b4', '#2ca02c', '#d62728',
                                                      '#17becf', '#8c564b', '#9467bd', '#e377c2'])
for i, (x, y) in enumerate(zip(xs, ys)):
    ax.text(i, y + .15, f'{y:.1f}', ha='center', fontsize=8)
ax.set_ylabel('forgetting (pp)')
ax.set_title('forgetting by full-credit subset (r=12.5%)')
plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)

# 2) B (hybrid) vs All by order
ax = axes[1]
orders = ['o0', 'o1', 'o2']
w = .35
for i, (arm, c) in enumerate([('B', '#1f77b4'), ('All', '#e377c2')]):
    ys = [np.mean([r['forgetting'] * 100 for r in rows if r['arm'] == arm and r['order'] == o
                   and r['ratio'] == .125]) for o in orders]
    ax.bar(np.arange(3) + (i - .5) * w, ys, width=w, label=arm, color=c, alpha=.85)
    for j, y in enumerate(ys):
        ax.text(j + (i - .5) * w, y + .1, f'{y:.1f}', ha='center', fontsize=8)
ax.set_xticks(range(3), orders)
ax.set_ylabel('forgetting (pp)')
ax.set_title('hybrid (B) vs full core credit (All), r=12.5%')
ax.legend(fontsize=8)

# 3) main effects with per-seed dots
ax = axes[2]
if summ and 'me125' in summ:
    for j, fac in enumerate(('W', 'route', 'hold')):
        dd = np.array(summ['me125']['me'][fac]['diffs']) * 100
        ax.scatter(np.full(len(dd), j) + np.linspace(-.08, .08, len(dd)), dd, s=14, alpha=.6)
        ax.bar([j], [dd.mean()], width=.45, alpha=.4, color='k')
        ax.text(j, dd.mean(), f'{dd.mean():+.2f}', ha='center', va='bottom', fontsize=9)
    ax.axhline(0, color='k', lw=.5)
    ax.set_xticks(range(3), ['W', 'route', 'hold'])
    ax.set_ylabel('ME (pp), positive = full credit helps')
    ax.set_title('main effect of 20-step credit (r=12.5%)')
else:
    ax.text(.5, .5, 'no summary.json', ha='center')
fig.tight_layout()
fig.savefig(ROOT / 'attribution.png', dpi=150)
plt.close(fig)

# 4) credit audit
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
p = ROOT / 'results/credit_audit.json'
audit = json.loads(p.read_text()) if p.exists() else []

ax = axes[0]
srcs = [s for s in ('init', 'single', 'cl_r0', 'cl_r12.5') if any(r['source'] == s for r in audit)]
width = .16
for i, g in enumerate(GROUPS):
    ys = []
    for src in srcs:
        vs = [r[f'cos5_{g}'] for r in audit if r['source'] == src]
        ys.append(np.nanmean(vs) if vs else np.nan)
    ax.bar(np.arange(len(srcs)) + (i - 2) * width, ys, width=width, label=g)
ax.set_xticks(range(len(srcs)), srcs)
ax.set_ylabel('cos(g5, g20)')
ax.set_title('credit direction loss by group / source')
ax.legend(fontsize=6, ncol=2)
ax.axhline(1, color='k', lw=.4)

ax = axes[1]
for src in ('cl_r12.5', 'cl_r0'):
    rs = [r for r in audit if r['source'] == src]
    if not rs:
        continue
    for g, c in zip(GROUPS, ['C0', 'C1', 'C2', 'C3', 'C4']):
        ys = []
        for st in range(4):
            vs = [r[f'cos5_{g}'] for r in rs if r.get('stage') == st]
            ys.append(np.nanmean(vs) if vs else np.nan)
        ax.plot(range(4), ys, marker='o', color=c, ls='-' if src == 'cl_r12.5' else '--',
                label=f'{g} {src}')
ax.set_xlabel('CL stage')
ax.set_ylabel('cos(g5, g20)')
ax.set_title('credit gap along continual learning')
ax.legend(fontsize=6, ncol=2)

ax = axes[2]
for arm, c in [('B', '#1f77b4'), ('All', '#e377c2')]:
    rs = [r for r in rows if r['arm'] == arm and r['ratio'] == .125 and r['seed'] in (11, 22, 33, 44, 55)]
    for g, ls in [('W', '-'), ('route', '--'), ('hold', ':')]:
        ys = []
        for st in range(4):
            vs = [sm['credit'][f'cos5_{g}'] for r in rs for sm in [r['stage_metrics'][st]] if sm.get('credit')]
            ys.append(np.nanmean(vs) if vs else np.nan)
        ax.plot(range(4), ys, marker='o', color=c, ls=ls, label=f'{arm}:{g}')
ax.set_xlabel('CL stage')
ax.set_ylabel('cos(g5, g20)')
ax.set_title('in-training credit gap (arms B vs All)')
ax.legend(fontsize=6, ncol=2)
fig.tight_layout()
fig.savefig(ROOT / 'credit_audit.png', dpi=150)
plt.close(fig)
print('wrote attribution.png, credit_audit.png')

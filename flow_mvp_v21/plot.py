"""第二十一轮绘图：B overlap 轨迹 / 状态子空间重叠 / 冲突-遗忘 / 机制链。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
PHASES = {'A': 'results/A', 'B': 'results/B', 'C': 'results/C'}
ARMS = ['fixed-overlap', 'fixed-disjoint', 'learnable', 'learnable+orth0.1',
        'learnable+overlap0.1', 'learnable+orth1', 'learnable+overlap1', 'learnable-random']


def cl_rows():
    rows = []
    for p in PHASES.values():
        d = ROOT / p
        for f in d.glob('*.json'):
            if f.name in ('config.json', 'h3.json') or f.name.startswith('metrics_'):
                continue
            r = json.loads(f.read_text())
            if 'matrix' in r:
                rows.append(r)
    return rows


rows = cl_rows()
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# 1) O_B 训练轨迹（stage 级，按 arm 平均）
ax = axes[0]
for arm in ['learnable', 'learnable+orth0.1', 'learnable+overlap0.1', 'learnable+overlap1']:
    rs = [r for r in rows if r['arm'] == arm and r['ratio'] == 0]
    if not rs:
        continue
    ys = []
    for r in rs:
        path = [r['B0']['O_B']] + [sm['O_B'] for sm in r['stage_metrics']]
        ys.append(path)
    y = np.mean(ys, 0)
    ax.plot(range(len(y)), y, marker='o', label=arm.replace('learnable', 'L'))
ax.set_xlabel('stage (B metrics after each task)')
ax.set_ylabel('O_B')
ax.set_title('B amplitude overlap across stages (r=0)')
ax.legend(fontsize=7)
ax.axhline(0, color='k', lw=.4)

# 2) O_h(t) 曲线（final stage，按 arm 平均）
ax = axes[1]
for arm in ARMS:
    rs = [r for r in rows if r['arm'] == arm and r['ratio'] == 0]
    rs = [r for r in rs if 'O_h_t2' in r['stage_metrics'][-1]]
    if not rs:
        continue
    y = [np.mean([r['stage_metrics'][-1][f'O_h_t{t}'] for r in rs]) for t in (2, 4, 8, 20)]
    ax.plot([2, 4, 8, 20], y, marker='o', label=arm)
ax.set_xlabel('t')
ax.set_ylabel('O_h(t)')
ax.set_title('counterfactual state subspace overlap (final stage, r=0)')
ax.legend(fontsize=6)

# 3) 冲突 C∇ vs forgetting（r=0，arm×seed）
ax = axes[2]
for arm in ARMS:
    xs, ys = [], []
    for seed in sorted({r['seed'] for r in rows if r['arm'] == arm}):
        rs = [r for r in rows if r['arm'] == arm and r['seed'] == seed and r['ratio'] == 0]
        if not rs:
            continue
        cs = []
        for r in rs:
            for sm in r['stage_metrics']:
                c = sm.get('conflict')
                if c and all(c.get(g) is not None for g in ('W', 'route', 'hold')):
                    cs.append(-np.mean([c['W'], c['route'], c['hold']]))
        if cs:
            xs.append(np.mean(cs))
            ys.append(np.mean([r['forgetting'] for r in rs]) * 100)
    ax.scatter(xs, ys, s=14, label=arm, alpha=.7)
ax.set_xlabel('C_grad = -mean cos(W,Route,Hold)')
ax.set_ylabel('forgetting (pp)')
ax.set_title('conflict vs forgetting (r=0, arm x seed)')
ax.legend(fontsize=6)

fig.tight_layout()
fig.savefig(ROOT / 'decoupling.png', dpi=150)
plt.close(fig)

# 4) D 阶段敏感性
fig, ax = plt.subplots(figsize=(7, 4))
dro = [json.loads(f.read_text()) for f in (ROOT / 'results/D').glob('*.json') if 'matrix' in f.read_text()]
for arm in sorted({r['arm'] for r in dro}):
    for ratio, st in ((0, '-o'), (.125, '--s')):
        rs = [r for r in dro if r['arm'] == arm and r['ratio'] == ratio]
        if rs:
            ax.bar(f'{arm}\nr={ratio}', np.mean([r['forgetting'] for r in rs]) * 100, alpha=.7)
ax.set_ylabel('forgetting (pp)')
ax.set_title('V21D Full-BPTT sensitivity (o0, r=0/12.5%)')
plt.setp(ax.get_xticklabels(), rotation=25, ha='right', fontsize=7)
fig.tight_layout()
fig.savefig(ROOT / 'fullbptt_sensitivity.png', dpi=150)
plt.close(fig)
print('wrote decoupling.png, fullbptt_sensitivity.png')

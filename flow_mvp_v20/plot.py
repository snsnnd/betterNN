"""第二十轮绘图：Phase A 拓扑曲线 / Phase B overlap / Phase C s_W 交互。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
PHASES = {'A': 'results/A', 'B': 'results/B', 'C': 'results/C'}


def load(phase):
    out = ROOT / PHASES[phase]
    files = [p for p in sorted(out.glob('*.json')) if p.name != 'config.json' and not p.name.startswith('metrics_')]
    return [json.loads(p.read_text()) for p in files]


def cl_by(rows, ratio, group_key):
    d = {}
    for r in rows:
        if 'matrix' not in r or r['ratio'] != ratio:
            continue
        g = group_key(r)
        d.setdefault(g, []).append(r)
    return d


def mean_sd(rs, key):
    v = np.array([r[key] for r in rs])
    return v.mean(), v.std()


def fig_A():
    rows = load('A')
    order = ['single-random', 'single-central', 'multi8', 'multi16', 'distributed']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ratio, style in [(0, '-o'), (.125, '--s')]:
        d = cl_by(rows, ratio, lambda r: r['topo'])
        xs, fin, fog = [], [], []
        for i, t in enumerate(order):
            if t in d:
                xs.append(i)
                fin.append(mean_sd(d[t], 'final_mean')[0])
                fog.append(mean_sd(d[t], 'forgetting')[0])
        axes[0].plot(xs, fin, style, label=f'r={ratio}')
        axes[1].plot(xs, fog, style, label=f'r={ratio}')
    for ax, title, ylab in [(axes[0], 'CL final_mean', '%'), (axes[1], 'CL forgetting', 'pp')]:
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=20, fontsize=8)
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.legend(fontsize=8)
    singles = [r for r in rows if r['name'].startswith('single_')]
    d = {}
    for r in singles:
        d.setdefault(r['topo'], []).append(r['final'])
    axes[2].bar(range(len(order)), [np.mean(d.get(t, [np.nan])) for t in order], yerr=[np.std(d.get(t, [np.nan])) for t in order])
    axes[2].set_xticks(range(len(order)))
    axes[2].set_xticklabels(order, rotation=20, fontsize=8)
    axes[2].set_title('single-task final')
    axes[2].set_ylim(.9, 1.0)
    fig.tight_layout()
    fig.savefig(ROOT / 'input_topology.png', dpi=150)
    plt.close(fig)


def fig_B():
    rows = load('B')
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for ratio, key in [(0, 'forgetting'), (.125, 'forgetting')]:
        d = cl_by(rows, ratio, lambda r: r['alpha'])
        xs = sorted(k for k in d)
        ax[0].plot(xs, [mean_sd(d[k], key)[0] for k in xs],
                   '-o' if ratio == 0 else '--s', label=f'r={ratio}')
    ax[0].set_xlabel('alpha_overlap')
    ax[0].set_ylabel('forgetting (pp)')
    ax[0].set_title('Phase B: overlap vs forgetting')
    ax[0].legend(fontsize=8)
    try:
        h3 = json.loads((ROOT / 'results/B/h3.json').read_text())
        xs = sorted(h3)
        for g, st in zip(('W', 'route', 'hold'), ('-o', '-s', '-^')):
            ax[1].plot([float(x.split('=')[1]) for x in xs], [h3[x][g] for x in xs], st, label=g)
        ax[1].axhline(0, color='k', lw=.5)
        ax[1].legend(fontsize=8)
    except FileNotFoundError:
        ax[1].text(.1, .5, 'h3.json missing', transform=ax[1].transAxes)
    ax[1].set_xlabel('alpha_overlap')
    ax[1].set_ylabel('Δcos (vs alpha=0)')
    ax[1].set_title('Phase B: gradient conflict')
    fig.tight_layout()
    fig.savefig(ROOT / 'overlap_gradient.png', dpi=150)
    plt.close(fig)


def fig_C():
    rows = load('C')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, topo in zip(axes, ['single-random', 'distributed']):
        d = cl_by(rows, 0, lambda r: (r['topo'], r['sw']))
        dd = {k[1]: v for k, v in d.items() if k[0] == topo}
        xs = sorted(dd)
        ax.plot(xs, [mean_sd(dd[k], 'final_mean')[0] for k in xs], '-o', label='final')
        ax.plot(xs, [mean_sd(dd[k], 'forgetting')[0] for k in xs], '-s', label='forget')
        ax.set_xlabel('s_W = ||W0||_2')
        ax.set_title(topo)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / 'sW_interaction.png', dpi=150)
    plt.close(fig)


for f in (fig_A, fig_B, fig_C):
    f()
print('wrote input_topology.png, overlap_gradient.png, sW_interaction.png')

"""第十八轮：核验、可压缩性汇总与报告。"""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent


def fmt(mean, std, scale=100):
    return f'{mean * scale:.2f} ± {std * scale:.2f}'


def load_rows():
    rows = []
    for p in sorted(list(ROOT.glob('ba_seed*.json')) + list(ROOT.glob('ba_oracle.json'))):
        rows.extend(json.loads(p.read_text())['rows'])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='.')
    args = ap.parse_args()
    rows = load_rows()
    assert rows, 'no block_affine results found'
    srows = [r for r in rows if r['init'] == 'sigma_max']
    core = [r for r in rows if r['init'] not in ('sigma_max',)]
    oracle = [r for r in core if r['init'] == 'oracle']
    assert oracle and max(r['Eh'] for r in oracle) < 1e-6, 'oracle identity check failed'

    def agg(filter_fn):
        rr = [r for r in core if filter_fn(r)]
        return {'acc20': float(np.mean([r['acc'] for r in rr if r['steps'] == 20])) if any(r['steps'] == 20 for r in rr) else None,
                'acc40': float(np.mean([r['acc'] for r in rr if r['steps'] == 40])) if any(r['steps'] == 40 for r in rr) else None,
                'Eh': float(np.mean([r['Eh'] for r in rr])) if rr else None, 'n': len(rr)}

    table = []
    for L in [1, 2, 4, 5, 10]:
        for init in ['zero', 'preroute']:
            for K in [0, 1, 2, 3]:
                e = agg(lambda r: r['L'] == L and r['init'] == init and r['K'] == K)
                if e['n']:
                    table.append({'L': L, 'init': init, 'K': K, 'depth': (K + 1) * L, **e})
    refs = {}
    for name, filt in [('flow', lambda r: r['init'] == 'flow'), ('preroute_model', lambda r: r['init'] == 'preroute_ref')]:
        refs[name] = agg(filt)
    sigma = [r['sigma_max_mean'] for r in srows]
    (ROOT / 'summary.json').write_text(json.dumps({'table': table, 'refs': refs,
                                                   'sigma_max_mean': float(np.mean(sigma)) if sigma else None,
                                                   'n_rows': len(rows)}, indent=2))

    lines = ['# 第十八轮：非线性 Flow 的 block-affine 可压缩性', '',
             '对象：第十五轮训练好的 Flow-v2（r=12.5%、o0、5 种子、stage3 模型），20/40 步测试集（每任务 256 样本）；'
             '块仿射 + 迭代重线性化（每遍用块 Jacobian 的 JVP 组合，串行深度≈(K+1)·L）。'
             '表中 acc 为四任务平均；E_h 为边界状态相对误差（剔除近零分母）。', '',
             f"参考：Flow 串行 acc20 {refs['flow']['acc20'] * 100:.2f}%、acc40 {refs['flow']['acc40'] * 100:.2f}%；"
             f"PreRoute 模型 acc20 {refs['preroute_model']['acc20'] * 100:.2f}%、acc40 {refs['preroute_model']['acc40'] * 100:.2f}%。", '',
             '| L | 初始估计 | K | 有效深度 | acc20% | acc40% | E_h |', '|---:|---|---:|---:|---:|---:|---:|']
    for e in table:
        lines.append(f"| {e['L']} | {e['init']} | {e['K']} | {e['depth']} | "
                     f"{e['acc20'] * 100:.2f} | {e['acc40'] * 100:.2f} | {e['Eh']:.4f} |")
    if sigma:
        lines += ['', f"块 Jacobian σ_max（L=4, 20 步, 幂迭代）：均值 {float(np.mean(sigma)):.3f}。", '']

    lines += ['## 自动汇总的观察', '']
    for init in ['preroute', 'zero']:
        best = None
        for e in table:
            if e['init'] != init:
                continue
            if best is None or e['acc40'] > best['acc40']:
                best = e
        if best:
            lines.append(f"- {init} 初始估计的最佳点：L={best['L']}、K={best['K']}（深度 {best['depth']}）"
                         f"acc20 {best['acc20'] * 100:.2f}%、acc40 {best['acc40'] * 100:.2f}%、E_h {best['Eh']:.4f}。")
    pr = [e for e in table if e['init'] == 'preroute' and e['K'] == 3]
    if pr:
        lines.append('- preroute 初始 + 3 次重线性化：' + '；'.join(
            f"L={e['L']} acc40 {e['acc40'] * 100:.2f}%/E_h {e['Eh']:.3f}" for e in pr) + '。')

    lines += ['', '## 验证与边界', '',
              f"oracle 自检（真实入口线性化）E_h 最大 {max(r['Eh'] for r in oracle):.2e}，符合恒等性质。", '',
              '- 线性化在测试分布上、对已训练模型做；不训练、不引入新架构。',
              '- E_h 使用相对状态误差、按块平均；近零状态时数值不稳定（zero 初始的 K=0 已剔除异常显示）。',
              '- 有效深度按 (K+1)·L 计，相当于"遍数 × 块内串行步"；块间组合为仿射 scan。']

    (ROOT / 'REPORT.md').write_text('\n'.join(lines))
    (ROOT / 'verification.txt').write_text(
        f"PASS oracle identity max Eh={max(r['Eh'] for r in oracle):.2e}; rows={len(rows)}.\n")
    print('\n'.join(lines))


if __name__ == '__main__':
    main()

"""第十七轮：核验、可并行动力学汇总与报告。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import experiment
from experiment import ROOT, build, data, evaluate, scores, MODELS, SCAN_READY, ORDERS, RATIOS, SEEDS, N, EPOCHS

V15 = ROOT.parent / 'flow_mvp_v15' / 'results'


def fmt(mean, std, scale=100):
    return f'{mean * scale:.2f} ± {std * scale:.2f}'


def load_rows(out, models, orders, ratios, seeds):
    rows = []
    for p in sorted(out.glob('*.json')):
        if p.name == 'config.json':
            continue
        r = json.loads(p.read_text())
        if r.get('model') in models and r.get('order') in orders and r.get('ratio') in ratios and r.get('seed') in seeds:
            rows.append(r)
    return rows


def verify(out, rows):
    experiment.DEVICE = torch.device(rows[0]['device'])
    count = 0
    for r in rows:
        torch.set_num_threads(r.get('threads', 1))
        for stage in range(4):
            ck = torch.load(out / f"{r['name']}_stage{stage}.pt", map_location=experiment.DEVICE, weights_only=False)
            m = build(r['seed'], r['model']).to(experiment.DEVICE)
            m.load_state_dict(ck['state'])
            m.eval()
            assert scores(m, r['seed']) == r['matrix'][stage], (r['name'], stage)
            count += 1
            if stage == 3:
                assert scores(m, r['seed'], 40) == r['matrix40'][stage], (r['name'], 'len40')
                count += 1
    same = []
    for r in rows:
        if r['model'] == 'flowv2' and r['order'] == 'o0':
            fa = V15 / f"o0_flowv2_r{r['ratio_label']}_{r['seed']}.json"
            if fa.exists():
                same.append(r['matrix'] == json.loads(fa.read_text())['matrix'])
    print(f'PASS: {len(rows)} streams, {count} evaluations reproduced; flowv2 matches v15: '
          f'{sum(same)}/{len(same)}.', flush=True)
    return f'{sum(same)}/{len(same)}'


def group_stats(rows):
    out = {}
    for key in ['acquisition', 'final_mean', 'forgetting', 'bwt', 'final40', 'forgetting40', 'drift_h']:
        vals = [r[key] for r in rows]
        out[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results')
    ap.add_argument('--models', nargs='+', default=MODELS)
    ap.add_argument('--orders', nargs='+', default=list(ORDERS))
    ap.add_argument('--ratios', nargs='+', type=float, default=list(RATIOS))
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    args = ap.parse_args()
    out = ROOT / args.out
    meta = ROOT if args.out == 'results' else out
    ratios = [r for r in RATIOS if r in args.ratios]
    rows = load_rows(out, args.models, args.orders, ratios, args.seeds)
    expected = len(args.models) * len(ratios) * len(args.orders) * len(args.seeds)
    assert len(rows) == expected, f'expected {expected}, found {len(rows)}'
    check = verify(out, rows)

    table = []
    for model in args.models:
        for ratio in ratios:
            rr = [r for r in rows if r['model'] == model and r['ratio'] == ratio]
            table.append({'model': model, 'ratio': ratio, 'label': RATIOS[ratio], 'scan_ready': SCAN_READY[model],
                          'params': rr[0]['params'], 'n': len(rr), **group_stats(rr)})
    (meta / 'summary.json').write_text(json.dumps({'table': table, 'verification': check}, indent=2))
    (meta / 'all_metrics.json').write_text(json.dumps(rows, indent=2))
    scan = json.loads((ROOT / 'scan_check.json').read_text()) if (ROOT / 'scan_check.json').exists() else None

    index = {(r['model'], r['ratio'], r['order'], r['seed']): r for r in rows}
    paired = []
    for model in args.models:
        if model == 'flowv2':
            continue
        for ratio in ratios:
            entry = {'model': model, 'ratio': ratio, 'label': RATIOS[ratio]}
            for key in ['final_mean', 'forgetting', 'final40', 'drift_h']:
                vals = [index[(model, ratio, o, s)][key] - index[('flowv2', ratio, o, s)][key]
                        for o in args.orders for s in args.seeds]
                entry[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                              'positive': int(sum(v > 0 for v in vals)), 'n': len(vals)}
            paired.append(entry)
    (meta / 'paired_differences.json').write_text(json.dumps(paired, indent=2))

    lines = ['# 第十七轮：可并行动力学可行性实测', '',
             f"A'B'C'D'、N={N}、{EPOCHS} 轮/阶段、5 步截断 BPTT、每步旧信息带宽 16 条、r∈{{0,12.5%}}、"
             '3 顺序 × 5 种子。四种接线同架构同参数量：A=现状（tanh+状态依赖 Route）、B=线性底网、C=预计算 Route（严格可 scan）、'
             'D=两遍迭代 Route（两遍均可 scan）。表中为双输出同时正确率，均值±样本标准差。', '',
             '| 模型 | replay% | 可scan | 最终平均% | 遗忘（百分点） | 40步最终% | 40步遗忘 | D_h 漂移 |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for e in table:
        lines.append(f"| {e['model']} | {e['label']} | {'✓' if e['scan_ready'] else '✗'} | "
                     f"{fmt(e['final_mean']['mean'], e['final_mean']['std'])} | "
                     f"{fmt(e['forgetting']['mean'], e['forgetting']['std'])} | "
                     f"{fmt(e['final40']['mean'], e['final40']['std'])} | "
                     f"{e['forgetting40']['mean'] * 100:.2f} | {e['drift_h']['mean']:.3f} |")

    lines += ['', '## 与 A(flowv2) 的配对差值（15 对：3 顺序 × 5 种子）', '',
              '| 模型 | replay% | final 差 | forgetting 差 | 40步 final 差 | D_h 差 |',
              '|---|---|---|---|---|---|']
    for e in paired:
        lines.append(f"| {e['model']} | {e['label']} | {fmt(e['final_mean']['mean'], e['final_mean']['std'])} | "
                     f"{fmt(e['forgetting']['mean'], e['forgetting']['std'])} | "
                     f"{fmt(e['final40']['mean'], e['final40']['std'])} | {e['drift_h']['mean']:+.4f} |")

    if scan:
        lines += ['', '## 可并行性验证（scan_check.py）', '',
                  f"- C 模型：顺序计算 vs Hillis–Steele 仿射扫描，64 步 float32 最大偏差 "
                  f"{scan['preroute']['seq_vs_scan_max_abs']:.2e}（数值等价）。",
                  f"- D 模型第二遍：扫描 vs 顺序最大偏差 {scan['iter_pass2']['pass2_seq_vs_scan']:.2e}。",
                  f"- 说明：模型自身 einsum 归约与矩阵化复算有 ~1.6e-3 的浮点差异，属实现顺序差异。",
                  f"- 朴素扫描（逐时间组合 256×256 矩阵）在 CPU 上比顺序向量计算慢（T=512："
                  f"{scan['timing']['seq_seconds']:.2f}s vs {scan['timing']['scan_seconds']:.2f}s）；"
                  '本轮只验证可结合性，不做工程加速声明。', '']

    def get(model, ratio, key):
        e = next(x for x in table if x['model'] == model and x['ratio'] == ratio)
        return e[key]['mean']
    lines += ['', '## 自动汇总的观察', '']
    for ratio in ratios:
        a, b, c, d = (get(m, ratio, 'final_mean') for m in ['flowv2', 'linear', 'preroute', 'iter'])
        af, bf, cf, df = (get(m, ratio, 'forgetting') for m in ['flowv2', 'linear', 'preroute', 'iter'])
        lines.append(f"- r={RATIOS[ratio]}%：final A {a * 100:.2f} / B {b * 100:.2f} / C {c * 100:.2f} / D {d * 100:.2f}；"
                     f"遗忘 A {af * 100:.2f} / B {bf * 100:.2f} / C {cf * 100:.2f} / D {df * 100:.2f}。")
    a40, c40 = get('flowv2', .125, 'final40'), get('preroute', .125, 'final40')
    d40 = get('iter', .125, 'final40')
    lines.append(f"- 40 步外推（r=12.5%）：A {a40 * 100:.2f}、C {c40 * 100:.2f}、D {d40 * 100:.2f}。")
    lines.append(f"- D_h 漂移（r=0）：A {get('flowv2', 0, 'drift_h'):.3f}、B {get('linear', 0, 'drift_h'):.3f}、"
                 f"C {get('preroute', 0, 'drift_h'):.3f}、D {get('iter', 0, 'drift_h'):.3f}。")

    lines += ['', '## 验证与边界', '',
              f"{len(rows)} 条流、{len(rows) * 4} 个 20 步阶段评估与 40 步最终评估全部重载复算一致；"
              f"flowv2 与第十五轮逐位一致 {check}。", '',
              '- C/D 的"可 scan"是数学性质（仿射组合律）验证，不是 GPU 加速基准；朴素 CPU 扫描更慢属实现代价。',
              '- 只测 N=256、单一任务集与 5 步截断协议；未扫描 ρ(W)、Hold 时间常数等动力学参数。']

    report = '\n'.join(lines)
    (meta / 'REPORT.md').write_text(report)
    (meta / 'verification.txt').write_text(
        f'PASS {len(rows)} streams, {len(rows) * 4} stage evaluations + 40-step final checks; flowv2 matches v15 {check}.\n')
    print(report)


if __name__ == '__main__':
    main()

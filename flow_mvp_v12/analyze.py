"""第十二轮：核验、等预算 replay 曲线、所需比例与报告。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import experiment
from experiment import ROOT, build, data, evaluate, scores, ORDERS, ARCHS, RATIOS, SEEDS, N, EPOCHS

V11 = ROOT.parent / 'flow_mvp_v11' / 'results'


def fmt(mean, std, scale=100):
    return f'{mean * scale:.2f} ± {std * scale:.2f}'


def load_rows(out, orders, archs, ratios, seeds):
    rows = []
    for p in sorted(out.glob('*.json')):
        if p.name == 'config.json':
            continue
        r = json.loads(p.read_text())
        if not isinstance(r, dict) or 'order' not in r or 'arch' not in r:
            continue
        if r['order'] in orders and r['arch'] in archs and r['ratio'] in ratios and r['seed'] in seeds:
            rows.append(r)
    return rows


def verify(out, rows):
    experiment.DEVICE = torch.device(rows[0]['device'])
    count = 0
    for r in rows:
        torch.set_num_threads(r.get('threads', 1))
        for stage in range(4):
            ck = torch.load(out / f"{r['name']}_stage{stage}.pt", map_location=experiment.DEVICE, weights_only=False)
            m = build(r['seed'], r['arch']).to(experiment.DEVICE)
            m.load_state_dict(ck['state'])
            m.eval()
            assert scores(m, r['seed']) == r['matrix'][stage], (r['name'], stage)
            count += 1
    print(f'PASS: {len(rows)} streams, {count} stage evaluations reproduced.', flush=True)


def group_stats(rows):
    out = {}
    for key in ['acquisition', 'final_mean', 'forgetting', 'bwt']:
        vals = [r[key] for r in rows]
        out[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}
    return out


def required_ratio(points, threshold):
    points = sorted(points)
    if points[0][1] <= threshold:
        return 0.0
    for i in range(1, len(points)):
        (x0, y0), (x1, y1) = points[i - 1], points[i]
        if y1 <= threshold:
            return x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results')
    ap.add_argument('--orders', nargs='+', default=list(ORDERS))
    ap.add_argument('--archs', nargs='+', default=ARCHS)
    ap.add_argument('--ratios', nargs='+', type=float, default=list(RATIOS))
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    args = ap.parse_args()
    out = ROOT / args.out
    meta = ROOT if args.out == 'results' else out
    ratios = [r for r in RATIOS if r in args.ratios]
    rows = load_rows(out, args.orders, args.archs, ratios, args.seeds)
    expected = len(args.orders) * len(args.archs) * len(ratios) * len(args.seeds)
    assert len(rows) == expected, f'expected {expected}, found {len(rows)}'
    verify(out, rows)

    pooled = {}
    per_order = {}
    for arch in args.archs:
        pooled[arch] = []
        for ratio in ratios:
            rr = [r for r in rows if r['arch'] == arch and r['ratio'] == ratio]
            pooled[arch].append({'ratio': ratio, 'label': RATIOS[ratio], 'n': len(rr), **group_stats(rr)})
    for order_key in args.orders:
        per_order[order_key] = []
        for arch in args.archs:
            for ratio in ratios:
                rr = [r for r in rows if r['order'] == order_key and r['arch'] == arch and r['ratio'] == ratio]
                per_order[order_key].append({'arch': arch, 'ratio': ratio, 'label': RATIOS[ratio], **group_stats(rr)})

    required = {}
    for arch in args.archs:
        points = [(e['ratio'], e['forgetting']['mean']) for e in pooled[arch]]
        required[arch] = {str(t): required_ratio(points, t) for t in [0.05, 0.03, 0.01]}
    (meta / 'summary.json').write_text(json.dumps({'pooled': pooled, 'per_order': per_order, 'required_ratio': required,
                                                   'params': {a: [r['params'] for r in rows if r['arch'] == a][0] for a in args.archs}}, indent=2))
    (meta / 'all_metrics.json').write_text(json.dumps(rows, indent=2))

    paired = []
    combos = [('flow', 'gru'), ('flow', 'rnn'), ('gru', 'rnn')]
    for a, b in combos:
        if a not in args.archs or b not in args.archs:
            continue
        for ratio in ratios:
            ra = [r for r in rows if r['arch'] == a and r['ratio'] == ratio]
            rb = [r for r in rows if r['arch'] == b and r['ratio'] == ratio]
            entry = {'a': a, 'b': b, 'ratio': ratio, 'label': RATIOS[ratio]}
            for key in ['acquisition', 'final_mean', 'forgetting']:
                vals = [x[key] - y[key] for x, y in zip(sorted(ra, key=lambda r: (r['order'], r['seed'])),
                                                        sorted(rb, key=lambda r: (r['order'], r['seed'])))]
                entry[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals, ddof=1)),
                              'positive': int(sum(v > 0 for v in vals)), 'values': vals}
            paired.append(entry)
    (meta / 'paired_differences.json').write_text(json.dumps(paired, indent=2))

    lines = ['# 第十二轮：Flow vs GRU vs RNN 等预算 replay 曲线', '',
             f"A'B'C'D' benchmark，每阶段 {EPOCHS} 轮；每步固定 128 样本，replay 按比例替换当前数据，"
             "每任务固定 32 个回放样本。三个顺序 × 5 种子。表中为双输出同时正确率，均值±样本标准差。", '',
             '| 架构 | 参数量 | replay% | acquisition% | 最终平均% | 遗忘百分点 |',
             '|---|---:|---:|---:|---:|---:|']
    params = {a: [r['params'] for r in rows if r['arch'] == a][0] for a in args.archs}
    for arch in args.archs:
        for e in pooled[arch]:
            lines.append(f"| {arch} | {params[arch]} | {e['label']} | " + ' | '.join(
                fmt(e[k]['mean'], e[k]['std']) for k in ['acquisition', 'final_mean', 'forgetting']) + ' |')

    lines += ['', '## 达到同样遗忘所需的最小 replay 比例（线性插值）', '',
              '| 架构 | 遗忘 ≤5pp | 遗忘 ≤3pp | 遗忘 ≤1pp |', '|---|---|---:|---:|']
    for arch in args.archs:
        r = required[arch]
        def show(v):
            return '0%（无需回放）' if v == 0.0 else ('未达到' if v is None else f'{v * 100:.2f}%')
        lines.append(f"| {arch} | {show(r['0.05'])} | {show(r['0.03'])} | {show(r['0.01'])} |")

    lines += ['', '## 架构配对差值（合并三个顺序，15 对）', '',
              '| 对比 | replay% | acquisition 差 | final_mean 差 | forgetting 差 |',
              '|---|---|---|---|---|']
    for e in paired:
        lines.append(f"| {e['a']} − {e['b']} | {e['label']} | {fmt(e['acquisition']['mean'], e['acquisition']['std'])} | "
                     f"{fmt(e['final_mean']['mean'], e['final_mean']['std'])} | "
                     f"{fmt(e['forgetting']['mean'], e['forgetting']['std'])} |")

    lines += ['', '## 各顺序的遗忘（百分点，均值）', '',
              '| 顺序 | 架构 | r=0 | r=6.25 | r=12.5 | r=25 |', '|---|---|---:|---:|---:|---:|']
    for order_key in args.orders:
        for arch in args.archs:
            cells = []
            for ratio in ratios:
                e = next(x for x in per_order[order_key] if x['arch'] == arch and x['ratio'] == ratio)
                cells.append(f"{e['forgetting']['mean'] * 100:.2f}")
            lines.append(f"| {order_key} | {arch} | " + ' | '.join(cells) + ' |')

    lines += ['', '## 自动汇总的观察', '']
    for arch in args.archs:
        fs = required[arch]
        def show(v):
            return '0' if v == 0.0 else ('未达到' if v is None else f'{v * 100:.2f}%')
        lines.append(f"- {arch}：达到遗忘≤5pp 需 {show(fs['0.05'])}、≤3pp 需 {show(fs['0.03'])}、≤1pp 需 {show(fs['0.01'])} replay。")
    if 'flow' in args.archs and 'gru' in args.archs and any(abs(r - .125) < 1e-9 for r in ratios):
        g = next(e for e in paired if e['a'] == 'flow' and e['b'] == 'gru' and abs(e['ratio'] - .125) < 1e-9)
        lines.append(f"- r=12.5% 时 flow − gru 的最终平均差 {g['final_mean']['mean'] * 100:+.2f}、遗忘差 {g['forgetting']['mean'] * 100:+.2f} 个百分点。")

    if V11.exists() and 'o0' in args.orders and 'flow' in args.archs:
        same = []
        for seed in args.seeds:
            for ratio in ratios:
                fa = out / f"o0_flow_r{RATIOS[ratio]}_{seed}.json"
                fb = V11 / f"o0_all_r{RATIOS[ratio]}_{seed}.json"
                if fa.exists() and fb.exists():
                    same.append(json.loads(fa.read_text())['matrix'] == json.loads(fb.read_text())['matrix'])
        if same:
            lines += ['', '## 与第十一轮的核对', '',
                      f"flow 路径与第十一轮 all 组的离散成绩矩阵一致：{sum(same)}/{len(same)} 条流。", '']

    lines += ['## 验证与边界', '',
              f"{len(rows)} 条流、{len(rows) * 4} 个阶段检查点全部重载复算一致。", '',
              '- 架构默认设置不同：flow 用 5 步截断 BPTT，GRU/RNN 用完整 BPTT；样本、更新次数、优化器与 replay 一致。',
              '- 隐藏宽度按参数量匹配，未对每个架构单独调参；只有 3 个顺序、5 个种子、单一规模与任务集。',
              '- 所需 replay 比例由合并均值线性插值得到，是描述性估计，不是统计检验。']

    report = '\n'.join(lines)
    (meta / 'REPORT.md').write_text(report)
    (meta / 'verification.txt').write_text(
        f'PASS {len(rows)} streams, {len(rows) * 4} stage evaluations reproduced.\n')
    print(report)


if __name__ == '__main__':
    main()

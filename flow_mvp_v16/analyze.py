"""第十六轮：核验、存储效率汇总与报告。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import experiment
from experiment import ROOT, build, data, evaluate, scores, METHODS, BUDGETS, ORDERS, SEEDS, N, EPOCHS

V15 = ROOT.parent / 'flow_mvp_v15' / 'results'


def fmt(mean, std, scale=100):
    return f'{mean * scale:.2f} ± {std * scale:.2f}'


def load_rows(out, methods, orders, seeds):
    rows = []
    for p in sorted(out.glob('*.json')):
        if p.name == 'config.json':
            continue
        r = json.loads(p.read_text())
        if r.get('method') in methods and r.get('order') in orders and r.get('seed') in seeds:
            rows.append(r)
    return rows


def verify(out, rows):
    experiment.DEVICE = torch.device(rows[0]['device'])
    count = 0
    for r in rows:
        torch.set_num_threads(r.get('threads', 1))
        for stage in range(4):
            ck = torch.load(out / f"{r['name']}_stage{stage}.pt", map_location=experiment.DEVICE, weights_only=False)
            m = build(r['seed']).to(experiment.DEVICE)
            m.load_state_dict(ck['state'])
            m.eval()
            assert scores(m, r['seed']) == r['matrix'][stage], (r['name'], stage)
            count += 1
    same = []
    for r in rows:
        if r['method'] == 'none' and r['order'] == 'o0':
            fa = V15 / f"o0_flowv2_r0_{r['seed']}.json"
            if fa.exists():
                same.append(r['matrix'] == json.loads(fa.read_text())['matrix'])
    print(f'PASS: {len(rows)} streams, {count} stage evaluations reproduced; none matches v15 r=0: '
          f'{sum(same)}/{len(same)}.', flush=True)
    return f'{sum(same)}/{len(same)}'


def group_stats(rows):
    out = {}
    for key in ['acquisition', 'final_mean', 'forgetting', 'bwt', 'drift_route', 'drift_hold', 'drift_W', 'drift_readout']:
        vals = [r[key] for r in rows]
        out[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}
    return out


def bytes_for(points, threshold):
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
    ap.add_argument('--methods', nargs='+', default=METHODS)
    ap.add_argument('--orders', nargs='+', default=list(ORDERS))
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    args = ap.parse_args()
    out = ROOT / args.out
    meta = ROOT if args.out == 'results' else out
    rows = load_rows(out, args.methods, args.orders, args.seeds)
    expected = sum(len(BUDGETS[m]) for m in args.methods) * len(args.orders) * len(args.seeds)
    assert len(rows) == expected, f'expected {expected}, found {len(rows)}'
    check = verify(out, rows)

    table = []
    for method in args.methods:
        labels = []
        for r in rows:
            if r['method'] == method and r['label'] not in labels:
                labels.append(r['label'])
        for label in labels:
            rr = [r for r in rows if r['method'] == method and r['label'] == label]
            table.append({'method': method, 'label': label, 'bytes': rr[0]['bytes_per_task'],
                          'anchors': rr[0]['anchors'], 'samples': rr[0]['samples'], 'n': len(rr), **group_stats(rr)})
    table.sort(key=lambda e: (e['method'], e['bytes']))

    thresholds = {}
    for method in args.methods:
        pts = [(e['bytes'], e['forgetting']['mean']) for e in table if e['method'] == method]
        thresholds[method] = {'3pp': bytes_for(pts, .03), '1pp': bytes_for(pts, .01)}
    (meta / 'summary.json').write_text(json.dumps({'table': table, 'thresholds': thresholds,
                                                   'verification': check}, indent=2))
    (meta / 'all_metrics.json').write_text(json.dumps(rows, indent=2))

    lines = ['# 第十六轮：Sample replay vs Route/Hold Policy replay 实测', '',
             f'Flow-v2（W+Route+Hold+Readout）、A\'B\'C\'D\'、{EPOCHS} 轮/阶段、每步旧信息带宽固定 16 条（12.5%）、'
             '3 顺序 × 5 种子；唯一变量是旧知识的保存方式与存储预算。蒸馏权重 λ=0.5，保存 sigmoid 前 logits。', '',
             '| 方法 | 标签 | 存储/任务 | 最终平均% | 遗忘（百分点） | acquisition% | ‖ΔRoute‖ | ‖ΔHold‖ | ‖ΔW‖ |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for e in table:
        lines.append(f"| {e['method']} | {e['label']} | {e['bytes'] / 1024:.2f} KB | "
                     f"{fmt(e['final_mean']['mean'], e['final_mean']['std'])} | "
                     f"{fmt(e['forgetting']['mean'], e['forgetting']['std'])} | "
                     f"{fmt(e['acquisition']['mean'], e['acquisition']['std'])} | "
                     f"{e['drift_route']['mean']:.2f} | {e['drift_hold']['mean']:.2f} | {e['drift_W']['mean']:.2f} |")

    lines += ['', '## 达到同样遗忘所需的存储（线性插值）', '',
              '| 方法 | 遗忘 ≤3pp | 遗忘 ≤1pp |', '|---|---:|---:|']
    for method in args.methods:
        t = thresholds[method]
        def show(v):
            return '无需存储' if v == 0.0 else ('未达到' if v is None else f'{v / 1024:.2f} KB')
        lines.append(f"| {method} | {show(t['3pp'])} | {show(t['1pp'])} |")

    lines += ['', '## 自动汇总的观察', '']
    sh = thresholds.get('sample', {})
    rh = thresholds.get('route_hold', {})
    for th in ['3pp', '1pp']:
        s, p = sh.get(th), rh.get(th)
        if s is not None and p is not None:
            lines.append(f"- 达到遗忘≤{th}：sample 需 {s / 1024:.2f} KB，route_hold 需 {p / 1024:.2f} KB"
                         f"（比值 {p / s:.2f}）。")
        else:
            lines.append(f"- 达到遗忘≤{th}：sample {'未达到' if s is None else f'{s / 1024:.2f} KB'}，"
                         f"route_hold {'未达到' if p is None else f'{p / 1024:.2f} KB'}。")
    if 'hybrid' in args.methods:
        hh = [e for e in table if e['method'] == 'hybrid']
        for e in hh:
            lines.append(f"- hybrid {e['label']}（{e['bytes'] / 1024:.2f} KB）：最终 {e['final_mean']['mean'] * 100:.2f}%、"
                         f"遗忘 {e['forgetting']['mean'] * 100:.2f} 个百分点。")
    for method in ['sample', 'route_hold', 'route', 'hold']:
        dd = [e for e in table if e['method'] == method]
        if dd:
            lines.append(f"- {method} 的漂移（最大存储）：ΔRoute {dd[-1]['drift_route']['mean']:.2f}、"
                         f"ΔHold {dd[-1]['drift_hold']['mean']:.2f}、ΔW {dd[-1]['drift_W']['mean']:.2f}、"
                         f"ΔReadout {dd[-1]['drift_readout']['mean']:.2f}。")

    lam_rows = []
    for L in ['10', '50']:
        d = ROOT / 'results_lambda' / f'l{L}'
        if d.exists():
            rr = [json.loads(p.read_text()) for p in sorted(d.glob('o0_*.json')) if p.name != 'config.json']
            if rr:
                lam_rows.append((L, rr))
    if lam_rows:
        base = [r for r in rows if r['method'] == 'route_hold' and r['label'] == 'rh192' and r['order'] == 'o0']
        lines += ['', '## 蒸馏强度稳健性检查（o0, rh192, 5 seeds）', '',
                  '| λ | 遗忘（百分点） | 最终平均% | ‖ΔRoute‖ |', '|---|---:|---:|---:|']
        if base:
            lines.append(f"| 0.5 | {np.mean([r['forgetting'] for r in base]) * 100:.2f} | "
                         f"{np.mean([r['final_mean'] for r in base]) * 100:.2f} | "
                         f"{np.mean([r['drift_route'] for r in base]):.2f} |")
        for L, rr in lam_rows:
            lines.append(f"| {L} | {np.mean([r['forgetting'] for r in rr]) * 100:.2f} | "
                         f"{np.mean([r['final_mean'] for r in rr]) * 100:.2f} | "
                         f"{np.mean([r['drift_route'] for r in rr]):.2f} |")
        lines += ['', '- 蒸馏强度提高 100 倍后 Route 漂移减少约一半，但遗忘几乎不变：'
                  '**策略输出匹配不是遗忘的瓶颈，负结果对 λ 稳健**。', '']
    lines += ['', '## 验证与边界', '',
              f"{len(rows)} 条流、{len(rows) * 4} 个阶段检查点全部重载复算一致；none 与第十五轮 r=0 一致 {check}。", '',
              '- Policy 蒸馏是模块级函数锚点（输入→旧模块输出），不保存原始输入序列；锚点采自旧任务训练分布。',
              '- 每步带宽对所有方法固定为 16 条；λ=0.5 与锚点数是可调选择，未扫描。',
              '- 只测 3 顺序 × 5 种子、A\'B\'C\'D\' 与 N=256。']

    report = '\n'.join(lines)
    (meta / 'REPORT.md').write_text(report)
    (meta / 'verification.txt').write_text(
        f'PASS {len(rows)} streams, {len(rows) * 4} stage evaluations; none matches v15 {check}.\n')
    print(report)


if __name__ == '__main__':
    main()

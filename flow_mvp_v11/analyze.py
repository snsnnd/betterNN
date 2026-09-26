"""第十一轮：核验、汇总、Pareto 与报告。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import experiment
from experiment import ROOT, build, data, evaluate, scores, ORDERS, ARMS, RATIOS, SEEDS, N, EPOCHS

V10 = ROOT.parent / 'flow_mvp_v10' / 'results' / 'cl'


def fmt(mean, std, scale=100):
    return f'{mean * scale:.2f} ± {std * scale:.2f}'


def load_rows(out, orders, arms, ratios, seeds):
    rows = []
    for p in sorted(out.glob('*.json')):
        if p.name == 'config.json':
            continue
        r = json.loads(p.read_text())
        if r['order'] in orders and r['arm'] in arms and r['ratio'] in ratios and r['seed'] in seeds:
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
        if r['arm'] == 'freeze_W':
            assert r['w_frozen_delta'] == 0, r['name']
    print(f'PASS: {len(rows)} streams, {count} stage evaluations reproduced; frozen W unchanged.', flush=True)


def group_stats(rows):
    keys = ['acquisition', 'final_mean', 'forgetting', 'bwt']
    out = {}
    for key in keys:
        vals = [r[key] for r in rows]
        out[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results')
    ap.add_argument('--orders', nargs='+', default=list(ORDERS))
    ap.add_argument('--arms', nargs='+', default=ARMS)
    ap.add_argument('--ratios', nargs='+', type=float, default=list(RATIOS))
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    args = ap.parse_args()
    out = ROOT / args.out
    meta = ROOT if args.out == 'results' else out
    ratios = [r for r in RATIOS if r in args.ratios]
    rows = load_rows(out, args.orders, args.arms, ratios, args.seeds)
    expected = len(args.orders) * len(args.arms) * len(ratios) * len(args.seeds)
    assert len(rows) == expected, f'expected {expected} streams, found {len(rows)}'
    verify(out, rows)

    per_order = {}
    for order_key in args.orders:
        per_order[order_key] = []
        for arm in args.arms:
            for ratio in ratios:
                rr = [r for r in rows if r['order'] == order_key and r['arm'] == arm and r['ratio'] == ratio]
                per_order[order_key].append({'arm': arm, 'ratio': ratio, 'label': RATIOS[ratio],
                                             'n': len(rr), **group_stats(rr)})
    pooled = []
    for arm in args.arms:
        for ratio in ratios:
            rr = [r for r in rows if r['arm'] == arm and r['ratio'] == ratio]
            pooled.append({'arm': arm, 'ratio': ratio, 'label': RATIOS[ratio], 'n': len(rr), **group_stats(rr)})
    (meta / 'summary.json').write_text(json.dumps({'per_order': per_order, 'pooled': pooled}, indent=2))
    (meta / 'all_metrics.json').write_text(json.dumps(rows, indent=2))

    paired = []
    for ratio in ratios:
        for order_key in args.orders + ['pooled']:
            if order_key == 'pooled':
                a = [r for r in rows if r['arm'] == 'freeze_W' and r['ratio'] == ratio]
                b = [r for r in rows if r['arm'] == 'all' and r['ratio'] == ratio]
            else:
                a = [r for r in rows if r['order'] == order_key and r['arm'] == 'freeze_W' and r['ratio'] == ratio]
                b = [r for r in rows if r['order'] == order_key and r['arm'] == 'all' and r['ratio'] == ratio]
            entry = {'ratio': ratio, 'label': RATIOS[ratio], 'order': order_key}
            for key in ['acquisition', 'final_mean', 'forgetting', 'bwt']:
                vals = [x[key] - y[key] for x, y in zip(sorted(a, key=lambda r: r['seed']), sorted(b, key=lambda r: r['seed']))]
                entry[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                              'positive': int(sum(v > 0 for v in vals)), 'values': vals}
            paired.append(entry)
    (meta / 'paired_differences.json').write_text(json.dumps(paired, indent=2))

    lines = ['# 第十一轮：Replay 比例 × 参数保护实测', '',
             f"N={N}，A'B'C'D' benchmark，每阶段 {EPOCHS} 轮；每步固定 128 样本，replay 按比例替换当前数据；"
             f"每个已学任务固定回放 {rows[0]['buffer_per_task']} 个样本。三个任务顺序，5 个种子，共 {len(rows)} 条流。"
             '表中为双输出同时正确率，均值±样本标准差（每个配置 5 个种子）。', '']

    for order_key in args.orders:
        lines += [f'## 顺序 {order_key} = {ORDERS[order_key]}', '',
                  '| 组 | replay% | acquisition% | 最终平均% | 遗忘百分点 | BWT百分点 |',
                  '|---|---:|---:|---:|---:|---:|']
        for e in per_order[order_key]:
            lines.append(f"| {e['arm']} | {e['label']} | " + ' | '.join(
                fmt(e[k]['mean'], e[k]['std']) for k in ['acquisition', 'final_mean', 'forgetting', 'bwt']) + ' |')
        lines.append('')

    lines += ['## 三个顺序合并', '',
              '| 组 | replay% | acquisition% | 最终平均% | 遗忘百分点 | BWT百分点 |',
              '|---|---:|---:|---:|---:|---:|']
    for e in pooled:
        lines.append(f"| {e['arm']} | {e['label']} | " + ' | '.join(
            fmt(e[k]['mean'], e[k]['std']) for k in ['acquisition', 'final_mean', 'forgetting', 'bwt']) + ' |')

    lines += ['', '## freeze_W − all 配对差值（合并三个顺序，15 对）', '',
              '| replay% | acquisition 差 | final_mean 差 | forgetting 差 | 正差值对数 |',
              '|---|---|---|---|---:|']
    for e in paired:
        if e['order'] != 'pooled':
            continue
        lines.append(f"| {e['label']} | {fmt(e['acquisition']['mean'], e['acquisition']['std'])} | "
                     f"{fmt(e['final_mean']['mean'], e['final_mean']['std'])} | "
                     f"{fmt(e['forgetting']['mean'], e['forgetting']['std'])} | "
                     f"{e['final_mean']['positive']}/{len(e['final_mean']['values'])} |")

    best_final = max(pooled, key=lambda e: e['final_mean']['mean'])
    best_forget = min(pooled, key=lambda e: e['forgetting']['mean'])
    best_acq = max(pooled, key=lambda e: e['acquisition']['mean'])
    lines += ['', '## 自动汇总的观察', '',
              f"- 最终平均最高：{best_final['arm']} × r={best_final['label']}%（{best_final['final_mean']['mean'] * 100:.2f}%），"
              f"其遗忘 {best_final['forgetting']['mean'] * 100:.2f} 个百分点、acquisition {best_final['acquisition']['mean'] * 100:.2f}%。",
              f"- 遗忘最低：{best_forget['arm']} × r={best_forget['label']}%（{best_forget['forgetting']['mean'] * 100:.2f} 个百分点），"
              f"其最终平均 {best_forget['final_mean']['mean'] * 100:.2f}%。",
              f"- acquisition 最高：{best_acq['arm']} × r={best_acq['label']}%（{best_acq['acquisition']['mean'] * 100:.2f}%）。"]
    for arm in args.arms:
        rr = [e for e in pooled if e['arm'] == arm]
        lines.append(f"- {arm} 随 replay 比例的最终平均：" + '；'.join(
            f"r={e['label']}% {e['final_mean']['mean'] * 100:.2f}" for e in rr) + '。')
        lines.append(f"- {arm} 随 replay 比例的遗忘：" + '；'.join(
            f"r={e['label']}% {e['forgetting']['mean'] * 100:.2f}" for e in rr) + '。')

    if V10.exists() and ('o0' in args.orders) and (0 in ratios):
        same = []
        for seed in args.seeds:
            fa = out / f'o0_all_r{RATIOS[0]}_{seed}.json'
            fb = V10 / f'{seed}.json'
            if fa.exists() and fb.exists():
                same.append(json.loads(fa.read_text())['matrix'] == json.loads(fb.read_text())['matrix'])
        if same:
            lines += ['', '## 与第十轮基线的核对', '',
                      f"o0 + all + r=0 与第十轮 CL 基线的离散成绩矩阵一致：{sum(same)}/{len(same)} 个种子。", '']

    lines += ['## 验证与边界', '',
              f"{len(rows)} 条流、{len(rows) * 4} 个阶段检查点全部重载复算一致；freeze_W 组的 W 在冻结后逐位不变。", '',
              '- replay 为等样本预算替换，比例越高当前任务暴露量越少；这既是机制也是代价。',
              '- 缓冲固定 32 样本/任务，未扫描缓冲大小；只有 3 个顺序、5 个种子、单一规模。',
              '- BWT/遗忘按各顺序的学习顺序计算，矩阵列始终是任务 ID。']

    report = '\n'.join(lines)
    (meta / 'REPORT.md').write_text(report)
    (meta / 'verification.txt').write_text(
        f'PASS {len(rows)} streams, {len(rows) * 4} stage evaluations; frozen W unchanged for freeze_W.\n')
    print(report)


if __name__ == '__main__':
    main()

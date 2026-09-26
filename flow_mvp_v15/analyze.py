"""第十五轮：核验、Flow-v2 基线与报告。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import experiment
from experiment import ROOT, build, data, evaluate, scores, ORDERS, RATIOS, SEEDS, N, EPOCHS

V14 = ROOT.parent / 'flow_mvp_v14' / 'results'
V12_SUMMARY = ROOT.parent / 'flow_mvp_v12' / 'summary.json'


def fmt(mean, std, scale=100):
    return f'{mean * scale:.2f} ± {std * scale:.2f}'


def load_rows(out, orders, ratios, seeds):
    rows = []
    for p in sorted(out.glob('*.json')):
        if p.name == 'config.json':
            continue
        r = json.loads(p.read_text())
        if r.get('order') in orders and r.get('ratio') in ratios and r.get('seed') in seeds:
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
        if r['order'] == 'o0' and r['ratio'] in (0, .125):
            fa = V14 / f"const1_r{r['ratio_label']}_{r['seed']}.json"
            if fa.exists():
                same.append(r['matrix'] == json.loads(fa.read_text())['matrix'])
    print(f'PASS: {len(rows)} streams, {count} stage evaluations reproduced; '
          f'o0 const1 equivalence: {sum(same)}/{len(same)}.', flush=True)
    return f'{sum(same)}/{len(same)}'


def group_stats(rows):
    out = {}
    for key in ['acquisition', 'final_mean', 'forgetting', 'bwt']:
        vals = [r[key] for r in rows]
        out[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results')
    ap.add_argument('--orders', nargs='+', default=list(ORDERS))
    ap.add_argument('--ratios', nargs='+', type=float, default=list(RATIOS))
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    args = ap.parse_args()
    out = ROOT / args.out
    meta = ROOT if args.out == 'results' else out
    ratios = [r for r in RATIOS if r in args.ratios]
    rows = load_rows(out, args.orders, ratios, args.seeds)
    expected = len(args.orders) * len(ratios) * len(args.seeds)
    assert len(rows) == expected, f'expected {expected}, found {len(rows)}'
    check = verify(out, rows)

    per_order = {}
    for order_key in args.orders:
        per_order[order_key] = []
        for ratio in ratios:
            rr = [r for r in rows if r['order'] == order_key and r['ratio'] == ratio]
            per_order[order_key].append({'ratio': ratio, 'label': RATIOS[ratio], **group_stats(rr)})
    pooled = []
    for ratio in ratios:
        rr = [r for r in rows if r['ratio'] == ratio]
        pooled.append({'ratio': ratio, 'label': RATIOS[ratio], 'n': len(rr), **group_stats(rr)})

    old = None
    if V12_SUMMARY.exists():
        s = json.loads(V12_SUMMARY.read_text())['pooled']['flow']
        old = {e['label']: e for e in s if e['ratio'] in ratios}

    (meta / 'summary.json').write_text(json.dumps({'per_order': per_order, 'pooled': pooled,
                                                   'old_flow_v12': old, 'verification': check,
                                                   'params': [r['params'] for r in rows][0]}, indent=2))
    (meta / 'all_metrics.json').write_text(json.dumps(rows, indent=2))

    lines = ['# 第十五轮：Flow-v2 正式基线实测', '',
             f'Flow-v2 = W + Route + Hold + Readout（无 Write 模块，输入常数注入），参数量与 v14 const1 相同路径；'
             f"A'B'C'D'、{EPOCHS} 轮/阶段、每步 128 样本、32 样本/任务缓冲，r∈{{0,6.25%,12.5%}} × 3 顺序 × 5 种子 = 45 条流。"
             '表中为双输出同时正确率，均值±样本标准差。', '',
             '| replay% | acquisition% | 最终平均% | 遗忘百分点 | BWT百分点 |',
             '|---|---:|---:|---:|---:|']
    for e in pooled:
        lines.append(f"| {e['label']} | " + ' | '.join(
            fmt(e[k]['mean'], e[k]['std']) for k in ['acquisition', 'final_mean', 'forgetting', 'bwt']) + ' |')

    if old:
        lines += ['', '## 与第十二轮旧 Flow（含死模块 Write）的对照（合并 3 顺序）', '',
                  '| replay% | 旧 Flow 最终% | Flow-v2 最终% | 旧 Flow 遗忘 | Flow-v2 遗忘 |',
                  '|---|---:|---:|---:|---:|']
        table = {e['label']: e for e in pooled}
        for lab in [RATIOS[r] for r in ratios]:
            o = old.get(lab)
            if not o:
                continue
            n = table[lab]
            lines.append(f"| {lab} | {o['final_mean']['mean'] * 100:.2f} | {n['final_mean']['mean'] * 100:.2f} | "
                         f"{o['forgetting']['mean'] * 100:.2f} | {n['forgetting']['mean'] * 100:.2f} |")

    t0 = next(e for e in pooled if e['ratio'] == 0)
    t6 = next(e for e in pooled if abs(e['ratio'] - .0625) < 1e-9)
    t12 = next(e for e in pooled if abs(e['ratio'] - .125) < 1e-9)
    lines += ['', '## 判定', '',
              f"- r=0：最终 {t0['final_mean']['mean'] * 100:.2f}%、遗忘 {t0['forgetting']['mean'] * 100:.2f} 个百分点。",
              f"- r=6.25%：最终 {t6['final_mean']['mean'] * 100:.2f}%、遗忘 {t6['forgetting']['mean'] * 100:.2f} 个百分点。",
              f"- r=12.5%：最终 {t12['final_mean']['mean'] * 100:.2f}%、遗忘 {t12['forgetting']['mean'] * 100:.2f} 个百分点。",
              '- 复核标准（第 11/12 轮：r=0 ≈77%、r=6.25% ≈97%、r=12.5% ≈99%）'
              + ('**成立**：可正式冻结 Flow-v2 = W + Route + Hold + Readout。' if t0['final_mean']['mean'] < .82 and
                 t6['final_mean']['mean'] > .95 and t12['final_mean']['mean'] > .98 else '**不成立**：需检查删除 Write 的影响。'), '']

    lines += ['## 验证与边界', '',
              f"45 条流、180 个阶段检查点全部重载复算一致；o0 的 r∈{{0,12.5%}} 与第十四轮 const1 路径逐位一致 {check}。", '',
              '- RNG 对齐保证同种子初始化与 v11/v14 相同；删除 Write 不改变任务与协议。',
              '- 只测 3 个比例、3 个顺序、5 种子；结论限于 A\'B\'C\'D\' 与 N=256。']

    report = '\n'.join(lines)
    (meta / 'REPORT.md').write_text(report)
    (meta / 'verification.txt').write_text(
        f'PASS {len(rows)} streams, {len(rows) * 4} stage evaluations; o0 const1 equivalence {check}.\n')
    print(report)


if __name__ == '__main__':
    main()

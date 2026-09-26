"""第十四轮：核验、Write 归因汇总与报告。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import experiment
from experiment import ROOT, build, data, evaluate, scores, WRITE_MODES, RATIOS, SEEDS, N, EPOCHS, AUX_LAMBDA

V11 = ROOT.parent / 'flow_mvp_v11' / 'results'
FREEZE_MODES = ['frozen', 'const1', 'const05']


def fmt(mean, std, scale=100):
    return f'{mean * scale:.2f} ± {std * scale:.2f}'


def load_rows(out, write_modes, ratios, seeds):
    rows = []
    for p in sorted(out.glob('*.json')):
        if p.name == 'config.json':
            continue
        r = json.loads(p.read_text())
        if r.get('write_mode') in write_modes and r.get('ratio') in ratios and r.get('seed') in seeds:
            rows.append(r)
    return rows


def verify(out, rows):
    experiment.DEVICE = torch.device(rows[0]['device'])
    count = 0
    for r in rows:
        torch.set_num_threads(r.get('threads', 1))
        for stage in range(4):
            ck = torch.load(out / f"{r['name']}_stage{stage}.pt", map_location=experiment.DEVICE, weights_only=False)
            m = build(r['seed'], r['write_mode']).to(experiment.DEVICE)
            m.load_state_dict(ck['state'])
            m.eval()
            assert scores(m, r['seed']) == r['matrix'][stage], (r['name'], stage)
            count += 1
        if r['write_mode'] in FREEZE_MODES:
            assert r['write_delta'] == 0, r['name']
        else:
            assert r['write_delta'] > 0, r['name']
    same = []
    for r in rows:
        if r['write_mode'] == 'frozen':
            fa = V11 / f"o0_all_r{r['ratio_label']}_{r['seed']}.json"
            if fa.exists():
                same.append(r['matrix'] == json.loads(fa.read_text())['matrix'])
    print(f'PASS: {len(rows)} streams, {count} stage evaluations reproduced; '
          f'frozen matches v11: {sum(same)}/{len(same)}.', flush=True)
    return {'frozen_vs_v11': f'{sum(same)}/{len(same)}'}


def group_stats(rows):
    out = {}
    for key in ['acquisition', 'final_mean', 'forgetting', 'bwt', 'write_delta']:
        vals = [r[key] for r in rows]
        out[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results')
    ap.add_argument('--write-modes', nargs='+', default=WRITE_MODES)
    ap.add_argument('--ratios', nargs='+', type=float, default=list(RATIOS))
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    args = ap.parse_args()
    out = ROOT / args.out
    meta = ROOT if args.out == 'results' else out
    ratios = [r for r in RATIOS if r in args.ratios]
    rows = load_rows(out, args.write_modes, ratios, args.seeds)
    expected = len(args.write_modes) * len(ratios) * len(args.seeds)
    assert len(rows) == expected, f'expected {expected}, found {len(rows)}'
    check = verify(out, rows)

    table = []
    for mode in args.write_modes:
        for ratio in ratios:
            rr = [r for r in rows if r['write_mode'] == mode and r['ratio'] == ratio]
            table.append({'write_mode': mode, 'ratio': ratio, 'label': RATIOS[ratio], 'n': len(rr), **group_stats(rr)})
    (meta / 'summary.json').write_text(json.dumps({'table': table, 'verification': check}, indent=2))
    (meta / 'all_metrics.json').write_text(json.dumps(rows, indent=2))

    index = {(r['write_mode'], r['ratio'], r['seed']): r for r in rows}
    paired = []
    for mode in args.write_modes:
        if mode == 'frozen':
            continue
        for ratio in ratios:
            entry = {'write_mode': mode, 'ratio': ratio, 'label': RATIOS[ratio]}
            for key in ['acquisition', 'final_mean', 'forgetting']:
                vals = [index[(mode, ratio, s)][key] - index[('frozen', ratio, s)][key] for s in args.seeds]
                entry[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                              'positive': int(sum(v > 0 for v in vals)), 'values': vals}
            paired.append(entry)
    (meta / 'paired_differences.json').write_text(json.dumps(paired, indent=2))

    lines = ['# 第十四轮：Write 因果归因实测', '',
             f"A'B'C'D'、Flow（N={N}）、o0、{EPOCHS} 轮/阶段、5 种子、r∈{{0,12.5%}}。唯一变量是写入门："
             'frozen（现状随机固定）、const1（p≡1）、const05（p≡0.5）、full_bptt（完整 BPTT 训练 Write）、'
             f'aux（5 步截断 + t=5 边界辅助 BCE，λ={AUX_LAMBDA}）。表中为双输出同时正确率，均值±样本标准差。', '',
             '| Write 模式 | replay% | acquisition% | 最终平均% | 遗忘百分点 | ‖ΔWrite‖ |',
             '|---|---:|---:|---:|---:|---:|']
    for e in table:
        lines.append(f"| {e['write_mode']} | {e['label']} | " + ' | '.join(
            fmt(e[k]['mean'], e[k]['std']) for k in ['acquisition', 'final_mean', 'forgetting'])
            + f" | {e['write_delta']['mean']:.3f} |")

    lines += ['', '## 与 frozen 的配对差值（5 种子）', '',
              '| 模式 | replay% | acquisition 差 | final_mean 差 | forgetting 差 | 正差值 |',
              '|---|---|---|---|---:|---:|']
    for e in paired:
        lines.append(f"| {e['write_mode']} | {e['label']} | {fmt(e['acquisition']['mean'], e['acquisition']['std'])} | "
                     f"{fmt(e['final_mean']['mean'], e['final_mean']['std'])} | "
                     f"{fmt(e['forgetting']['mean'], e['forgetting']['std'])} | {e['final_mean']['positive']}/5 |")

    lines += ['', '## 自动汇总的观察', '']
    for ratio in ratios:
        fr = next(e for e in table if e['write_mode'] == 'frozen' and e['ratio'] == ratio)
        c1 = next(e for e in table if e['write_mode'] == 'const1' and e['ratio'] == ratio)
        c05 = next(e for e in table if e['write_mode'] == 'const05' and e['ratio'] == ratio)
        lines.append(f"- r={RATIOS[ratio]}%：frozen 最终 {fr['final_mean']['mean'] * 100:.2f}%、遗忘 {fr['forgetting']['mean'] * 100:.2f}；"
                     f"const1 {c1['final_mean']['mean'] * 100:.2f}%/{c1['forgetting']['mean'] * 100:.2f}；"
                     f"const05 {c05['final_mean']['mean'] * 100:.2f}%/{c05['forgetting']['mean'] * 100:.2f}。")
    for e in paired:
        lines.append(f"- {e['write_mode']} − frozen（r={e['label']}%）：final {e['final_mean']['mean'] * 100:+.2f}、"
                     f"forgetting {e['forgetting']['mean'] * 100:+.2f} 个百分点，正差值 {e['final_mean']['positive']}/5。")

    fr = next(e for e in table if e['write_mode'] == 'frozen' and e['ratio'] == 0)
    c1 = next(e for e in table if e['write_mode'] == 'const1' and e['ratio'] == 0)
    ft = next(e for e in table if e['write_mode'] == 'full_bptt' and e['ratio'] == 0)
    ax = next(e for e in table if e['write_mode'] == 'aux' and e['ratio'] == 0)
    gap_const = abs(c1['final_mean']['mean'] - fr['final_mean']['mean']) * 100
    gap_full = ft['final_mean']['mean'] * 100 - fr['final_mean']['mean'] * 100
    gap_aux = ax['final_mean']['mean'] * 100 - fr['final_mean']['mean'] * 100
    verdict = 'const 与 frozen 接近 → 随机 task-conditioned 调制无明显贡献，可考虑删除 Write' if gap_const < 1 \
        else 'const 与 frozen 有明显差距 → 随机写入调制仍有结构贡献'
    lines += ['', f"- r=0 判定：|const1 − frozen| = {gap_const:.2f} 个百分点；full_bptt − frozen = {gap_full:+.2f}；"
              f"aux − frozen = {gap_aux:+.2f}。{verdict}。", '']

    lines += ['## 验证与边界', '',
              f"{len(rows)} 条流、{len(rows) * 4} 个阶段检查点全部重载复算一致；"
              f"冻结/常数模式 ΔWrite=0，训练模式 ΔWrite>0；frozen 与第十一轮一致 {check['frozen_vs_v11']}。", '',
              '- full_bptt 同时改变整网梯度协议，不只是 Write；aux 的 λ 与边界位置是可调选择。',
              '- 只测 o0 顺序与单一规模，结论限于本 benchmark。']

    report = '\n'.join(lines)
    (meta / 'REPORT.md').write_text(report)
    (meta / 'verification.txt').write_text(
        f'PASS {len(rows)} streams, {len(rows) * 4} stage evaluations; frozen matches v11 {check["frozen_vs_v11"]}.\n')
    print(report)


if __name__ == '__main__':
    main()

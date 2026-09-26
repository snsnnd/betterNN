"""重载第九轮检查点、复算、汇总并生成报告。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import experiment
from experiment import ROOT, build, data, evaluate, scores, NAME_TO_GROUP, TRAINABLE, ARMS, SEEDS, N, EPOCHS

V8 = ROOT.parent / 'flow_mvp_v8' / 'results'


def fmt(mean, std, scale=100):
    return f'{mean * scale:.2f} ± {std * scale:.2f}'


def verify(out, pre_rows, rows, controls):
    experiment.DEVICE = torch.device(rows[0]['device'])
    for r in pre_rows:
        torch.set_num_threads(r.get('threads', 1))
        m = build(r['seed'], 'all').to(experiment.DEVICE)
        ck = torch.load(out / 'pretrain' / f"{r['seed']}.pt", map_location=experiment.DEVICE, weights_only=False)
        m.load_state_dict(ck['state'])
        m.eval()
        assert scores(m, r['seed']) == r['matrix'][0], f"pretrain {r['seed']}"
    verified = 0
    for r in rows:
        pre_state = torch.load(out / 'pretrain' / f"{r['seed']}.pt", map_location='cpu', weights_only=False)['state']
        for stage in [1, 2, 3]:
            m = build(r['seed'], r['arm'])
            m.load_state_dict(pre_state)
            m = m.to(experiment.DEVICE)
            ck = torch.load(out / f"{r['arm']}_{r['seed']}_stage{stage}.pt",
                            map_location=experiment.DEVICE, weights_only=False)
            m.load_state_dict(ck['state'])
            m.eval()
            assert scores(m, r['seed']) == r['matrix'][stage], (r['arm'], r['seed'], stage)
            for n, p in m.named_parameters():
                g = NAME_TO_GROUP.get(n.split('.')[0])
                if g and g not in TRAINABLE[r['arm']]:
                    assert torch.equal(p.detach().cpu(), pre_state[n]), (r['arm'], stage, n)
            verified += 1
    for c in controls:
        m = build(c['seed'], 'all')
        m.load_state_dict(torch.load(out / 'pretrain' / f"{c['seed']}.pt", map_location='cpu', weights_only=False)['state'])
        m = m.to(experiment.DEVICE)
        ck = torch.load(out / f"control_{c['seed']}_{c['task']}.pt", map_location=experiment.DEVICE, weights_only=False)
        m.load_state_dict(ck['state'])
        m.eval()
        assert scores(m, c['seed']) == c['scores'], (c['seed'], c['task'])
    print(f'PASS: {len(pre_rows)} pretrain, {verified} stage checkpoints, {len(controls)} controls; '
          'frozen groups bitwise unchanged from pretrain.', flush=True)


def load(out, seeds, arms):
    pre_rows = [json.loads(p.read_text()) for p in sorted((out / 'pretrain').glob('*.json'))]
    rows = []
    for arm in arms:
        for s in seeds:
            f = out / f'{arm}_{s}.json'
            if f.exists():
                rows.append(json.loads(f.read_text()))
    controls = [json.loads(p.read_text()) for p in sorted(out.glob('control_*.json'))]
    assert len(pre_rows) == len(seeds), len(pre_rows)
    assert len(rows) == len(arms) * len(seeds), len(rows)
    assert len(controls) == len(seeds) * 3, len(controls)
    return pre_rows, rows, controls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results')
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--arms', nargs='+', default=ARMS)
    args = ap.parse_args()
    out = ROOT / args.out
    meta = ROOT if args.out == 'results' else out
    seeds, arms = args.seeds, args.arms
    pre_rows, rows, controls = load(out, seeds, arms)
    verify(out, pre_rows, rows, controls)

    pre_matrix = np.mean([r['matrix'][0] for r in pre_rows], axis=0)
    summary = []
    for arm in arms:
        rr = [r for r in rows if r['arm'] == arm]
        row = {'arm': arm, 'trainable': rr[0]['trainable'],
               'matrix': np.mean([r['matrix'] for r in rr], axis=0).tolist()}
        for key in ['acquisition_A', 'acquisition_BCD', 'final_mean', 'forgetting', 'bwt']:
            row[key] = {'mean': float(np.mean([r[key] for r in rr])),
                        'std': float(np.std([r[key] for r in rr], ddof=1))}
        for g in ['W', 'ctrl', 'readout']:
            vals = [r['group_delta'][g] for r in rr]
            row['delta_' + g] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals, ddof=1))}
        for stage in ['1', '2', '3']:
            reached = [r['epochs_to_90'][stage] for r in rr if r['epochs_to_90'][stage] is not None]
            row['to90_' + stage] = {'reached': len(reached), 'mean': float(np.mean(reached)) if reached else None}
        row['seconds'] = {'mean': float(np.mean([r['seconds'] for r in rr])),
                          'std': float(np.std([r['seconds'] for r in rr], ddof=1))}
        summary.append(row)
    (meta / 'summary.json').write_text(json.dumps(summary, indent=2))
    (meta / 'all_metrics.json').write_text(json.dumps(rows, indent=2))

    ctrl_summary = {}
    for task in [1, 2, 3]:
        cc = [c for c in controls if c['task'] == task]
        vals = [c['scores'][task] for c in cc]
        reached = [c['epochs_to_90'] for c in cc if c['epochs_to_90'] is not None]
        ctrl_summary[str(task)] = {'score_mean': float(np.mean(vals)), 'score_std': float(np.std(vals, ddof=1)),
                                   'reached': len(reached),
                                   'epochs_mean': float(np.mean(reached)) if reached else None}
    (meta / 'controls_summary.json').write_text(json.dumps(ctrl_summary, indent=2))

    index = {(r['arm'], r['seed']): r for r in rows}
    paired = []
    for arm in arms:
        if arm == 'all':
            continue
        entry = {'arm': arm}
        for key in ['acquisition_BCD', 'final_mean', 'forgetting', 'bwt']:
            vals = [index[(arm, s)][key] - index[('all', s)][key] for s in seeds]
            entry[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals, ddof=1)),
                          'positive': int(sum(v > 0 for v in vals)), 'values': vals}
        paired.append(entry)
    (meta / 'paired_differences.json').write_text(json.dumps(paired, indent=2))

    lines = ['# 第九轮：持续学习遗忘定位实测', '',
             f'N={N} dense。先用全部参数学习 A 作为共同起点，再依次学习 B、C、D；每组只允许指定参数组更新。'
             f'每阶段 {EPOCHS} 轮、无测试集选模型。replay 为等样本预算：每步 96 个当前任务 + 32 个旧任务样本，'
             '每个已学任务固定保留 32 个样本，当前任务暴露量比其他组少 25%。表中为双输出同时正确率，均值±样本标准差。', '',
             f"预训练 A 后平均四任务成绩：A {pre_matrix[0] * 100:.2f}，B {pre_matrix[1] * 100:.2f}，"
             f"C {pre_matrix[2] * 100:.2f}，D {pre_matrix[3] * 100:.2f}。", '',
             '## 各组主结果', '',
             '| 组 | 可训练参数 | acquisition_BCD% | 最终平均% | 遗忘百分点 | BWT百分点 | ‖ΔW‖ | ‖Δctrl‖ | ‖Δreadout‖ |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['arm']} | {r['trainable']} | "
                     + ' | '.join(fmt(r[k]['mean'], r[k]['std']) for k in ['acquisition_BCD', 'final_mean', 'forgetting', 'bwt'])
                     + f" | {r['delta_W']['mean']:.2f} | {r['delta_ctrl']['mean']:.2f} | {r['delta_readout']['mean']:.2f} |")

    lines += ['', '## 与全量更新（all）的配对差值', '',
              '| 组 | acquisition_BCD 差均值±SD | final_mean 差均值±SD | forgetting 差均值±SD |',
              '|---|---|---|---|']
    for e in paired:
        lines.append(f"| {e['arm']} | " + ' | '.join(fmt(e[k]['mean'], e[k]['std']) for k in
                                                      ['acquisition_BCD', 'final_mean', 'forgetting']) + ' |')

    lines += ['', '## 独立适应对照（从同一预训练起点单独学习该任务，全部参数）', '',
              '| 任务 | 单独学习后准确率% | 达到90%的种子数 | 达到者平均轮数 |',
              '|---|---:|---:|---:|']
    names = {1: 'B', 2: 'C', 3: 'D'}
    for task in [1, 2, 3]:
        c = ctrl_summary[str(task)]
        lines.append(f"| {names[task]} | {fmt(c['score_mean'], c['score_std'])} | {c['reached']}/5 | "
                     + (f"{c['epochs_mean']:.1f}" if c['epochs_mean'] else '—') + ' |')

    lines += ['', '## 新任务学习速度（验证达到90%的种子数/达到者平均轮数）', '',
              '| 组 | B | C | D |', '|---|---:|---:|---:|']
    for r in summary:
        cells = []
        for stage in ['1', '2', '3']:
            t = r['to90_' + stage]
            cells.append(f"{t['reached']}/5" + (f"，{t['mean']:.1f}轮" if t['mean'] else ''))
        lines.append(f"| {r['arm']} | " + ' | '.join(cells) + ' |')

    lines += ['', '## 各组平均任务×阶段矩阵', '']
    for r in summary:
        lines += [f"### {r['arm']}", '', '| 阶段结束 | A | B | C | D |', '|---|---|---|---|---|']
        labels = ['预训练A', '学完B', '学完C', '学完D']
        for i, row in enumerate(r['matrix']):
            lines.append(f"| {labels[i]} | " + ' | '.join(f'{v * 100:.2f}' for v in row) + ' |')
        lines.append('')

    armset = {r['arm'] for r in summary}
    obs = ['## 自动汇总的观察', '']
    if 'all' in armset:
        best_forget = min(summary, key=lambda r: r['forgetting']['mean'])
        best_final = max(summary, key=lambda r: r['final_mean']['mean'])
        all_row = next(r for r in summary if r['arm'] == 'all')
        freeze_rows = {r['arm']: r for r in summary if r['arm'].startswith('freeze_')}
        control_bcd = float(np.mean([ctrl_summary[str(t)]['score_mean'] for t in [1, 2, 3]]))
        obs += [f"- all 组：acquisition_BCD {all_row['acquisition_BCD']['mean'] * 100:.2f}%，最终平均 "
                f"{all_row['final_mean']['mean'] * 100:.2f}%，遗忘 {all_row['forgetting']['mean'] * 100:.2f} 个百分点。",
                f"- 遗忘最低的组：{best_forget['arm']}（{best_forget['forgetting']['mean'] * 100:.2f} 个百分点），"
                f"其 acquisition_BCD 为 {best_forget['acquisition_BCD']['mean'] * 100:.2f}%；最终平均最高的组："
                f"{best_final['arm']}（{best_final['final_mean']['mean'] * 100:.2f}%）。"]
        if freeze_rows:
            obs.append('- 三个冻结组与 all 的 forgetting 差值：'
                       + '；'.join(f"{arm} {freeze_rows[arm]['forgetting']['mean'] * 100 - all_row['forgetting']['mean'] * 100:+.2f}"
                                   for arm in freeze_rows) + ' 个百分点。')
        if {'ctrl_only', 'readout_only'} <= armset:
            obs.append('- ctrl_only 与 readout_only 的 acquisition_BCD：'
                       + '、'.join(f"{a} {next(r for r in summary if r['arm'] == a)['acquisition_BCD']['mean'] * 100:.2f}%"
                                   for a in ['ctrl_only', 'readout_only'])
                       + f"；独立适应对照 B/C/D 平均 {control_bcd * 100:.2f}%。若保护组远低于对照，说明该保护妨碍了新任务学习。")
        if any(e['arm'] == 'replay' for e in paired):
            e = next(e for e in paired if e['arm'] == 'replay')
            obs.append('- replay 与 all 的最终平均差值为 '
                       f"{e['final_mean']['mean'] * 100:+.2f}，遗忘差值 {e['forgetting']['mean'] * 100:+.2f} 个百分点；"
                       '注意其当前任务暴露量少 25%。')
    obs.append('')
    lines += obs

    if V8.exists():
        same = []
        for s in seeds:
            fa = out / f'all_{s}.json'
            fb = V8 / f'dense_{N}_{s}.json'
            if fa.exists() and fb.exists():
                same.append(json.loads(fa.read_text())['matrix'] == json.loads(fb.read_text())['matrix'])
        if same:
            lines += ['## 与第八轮 dense 的关系', '',
                      f"v9 all 组与第八轮 dense 的离散成绩矩阵一致：{sum(same)}/{len(same)} 个种子"
                      '（同设备、同协议，用于确认保护框架未改变基线路径）。', '']

    lines += ['## 验证', '',
              f"{len(pre_rows)} 个预训练检查点、{len(rows)} 条组结果、{len(controls)} 个独立适应对照全部重载；"
              '每个阶段的四任务成绩复算一致；所有被冻结参数组在阶段检查点中与预训练起点逐位相同。', '',
              '## 边界', '',
              '- 可训练自由度不同，冻结组与 all 的差异不能全部归因于“该参数组本身”。',
              '- replay 采用等样本预算替换，当前任务每轮少看 25% 数据；其收益与代价都要计入。',
              '- 只有一个任务顺序、5 个种子、单一合成任务与固定超参数；未测回放比例、任务顺序与更大规模。']

    report = '\n'.join(lines)
    (meta / 'REPORT.md').write_text(report)
    (meta / 'verification.txt').write_text(
        f'PASS {len(pre_rows)} pretrain, {len(rows)} arm rows, {len(controls)} controls; '
        'all stage scores reproduced; frozen groups bitwise unchanged.\n')
    print(report)


if __name__ == '__main__':
    main()

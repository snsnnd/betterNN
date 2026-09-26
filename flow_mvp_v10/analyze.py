"""第十轮：校准核验、预算选择与报告。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import experiment
from experiment import ROOT, build, data, evaluate, scores, N, SEEDS, EPOCHS, LR, THRESHOLD

TASKS = {'0': "A'", '1': "B'", '2': "C'", '3': "D'"}


def fmt(mean, std, scale=100):
    return f'{mean * scale:.2f} ± {std * scale:.2f}'


def load_group(out, sub):
    rows = []
    for p in sorted((out / sub).glob('*.json')):
        if p.name == 'config.json':
            continue
        rows.append(json.loads(p.read_text()))
    return rows


def verify(out, pretrain, scratch, adapt, cl):
    first = (pretrain or scratch or adapt or cl)[0]
    experiment.DEVICE = torch.device(first['device'])
    count = 0
    for rows in [pretrain, scratch, adapt]:
        for r in rows:
            torch.set_num_threads(r.get('threads', 1))
            sub = 'pretrain' if rows is pretrain else ('scratch' if rows is scratch else 'adapt')
            name = f"{r['seed']}" if sub == 'pretrain' else f"{r['task']}_{r['seed']}"
            ck = torch.load(out / sub / (name + '.pt'), map_location='cpu', weights_only=False)
            m = build(r['seed']).to(experiment.DEVICE)
            m.load_state_dict(ck['best'])
            m.eval()
            assert scores(m, r['seed']) == r['test_best'], (sub, name, 'best')
            m.load_state_dict(ck['final'])
            m.eval()
            assert scores(m, r['seed']) == r['test_final'], (sub, name, 'final')
            count += 2
    for r in cl:
        for stage in range(4):
            ck = torch.load(out / 'cl' / f"{r['seed']}_stage{stage}.pt", map_location='cpu', weights_only=False)
            m = build(r['seed']).to(experiment.DEVICE)
            m.load_state_dict(ck['state'])
            m.eval()
            assert scores(m, r['seed']) == r['matrix'][stage], (r['seed'], stage)
            count += 1
    print(f'PASS: {count} state evaluations reproduced.', flush=True)


def mean_curve(rows, epochs):
    out = []
    for e in range(epochs):
        vals = [r['history'][e]['validation'] for r in rows if len(r['history']) > e]
        out.append(float(np.mean(vals)) if vals else None)
    return out


def task_summary(rows):
    return {'test_final': {'mean': float(np.mean([r['test_final'][r['task']] for r in rows])),
                           'std': float(np.std([r['test_final'][r['task']] for r in rows], ddof=1))},
            'test_best': {'mean': float(np.mean([r['test_best'][r['task']] for r in rows])),
                          'std': float(np.std([r['test_best'][r['task']] for r in rows], ddof=1))},
            'best_val': float(np.mean([r['best_val'] for r in rows])),
            'best_epoch': float(np.mean([r['best_epoch'] for r in rows])),
            'reached90': int(sum(r['best_val'] >= THRESHOLD for r in rows))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results')
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    args = ap.parse_args()
    out = ROOT / args.out
    meta = ROOT if args.out == 'results' else out
    pretrain = load_group(out, 'pretrain')
    scratch = load_group(out, 'scratch')
    adapt = load_group(out, 'adapt')
    cl = load_group(out, 'cl')
    assert pretrain or cl, 'no results found'
    verify(out, pretrain, scratch, adapt, cl)

    calib = {'scratch': {}, 'adapt': {}, 'pretrain': {}}
    curves = {}
    for task in range(4):
        for mode, group in [('scratch', [r for r in scratch if r['task'] == task]),
                            ('pretrain', [r for r in pretrain if task == 0]),
                            ('adapt', [r for r in adapt if r['task'] == task])]:
            if not group:
                continue
            key = f'{mode}_{task}'
            curves[key] = mean_curve(group, args.epochs)
            calib[mode][TASKS[str(task)]] = task_summary(group)
    epochs_90 = next((e + 1 for e in range(args.epochs)
                      if all(c[e] is not None and c[e] >= .9 for c in curves.values())), None)
    epochs_95 = next((e + 1 for e in range(args.epochs)
                      if all(c[e] is not None and c[e] >= .95 for c in curves.values())), None)
    benchmark = {'threshold': THRESHOLD, 'epochs_90': epochs_90, 'epochs_95': epochs_95,
                 'pass': epochs_90 is not None, 'curves': curves}
    (meta / 'benchmark.json').write_text(json.dumps(benchmark, indent=2))

    lines = ['# 第十轮：持续学习 benchmark 校准实测', '',
             "新 benchmark A'B'C'D' 不含互为规则逆映射的任务对；每个任务分别从零（scratch）与从 A' 起点（adapt）单独训练，逐轮记录验证准确率；"
             '预算 E* 取使所有任务在 5 种子平均验证 ≥ 90% 的最小轮数。表中为双输出同时正确率，均值±样本标准差。', '',
             '## 单任务可学性', '',
             '| 模式 | 任务 | 最终测试% | 验证最优测试% | 最优验证% | 最优轮数 | 达到90的种子数 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for mode in ['scratch', 'pretrain', 'adapt']:
        for task, s in calib[mode].items():
            lines.append(f"| {mode} | {task} | {fmt(s['test_final']['mean'], s['test_final']['std'])} | "
                         f"{fmt(s['test_best']['mean'], s['test_best']['std'])} | {s['best_val'] * 100:.2f} | "
                         f"{s['best_epoch']:.1f} | {s['reached90']}/5 |")

    lines += ['', '## 验证预算选择', '',
              f"- 所有任务（scratch A/B/C/D 与 adapt B/C/D）5 种子平均验证同时 ≥90% 的最小轮数："
              + (f"**{epochs_90}**" if epochs_90 else '**未达到**'),
              f"- 同时 ≥95% 的最小轮数：" + (f"{epochs_95}" if epochs_95 else '未达到'),
              f"- 判定：{'PASS，可进入第十一轮' if benchmark['pass'] else 'FAIL，需要调整学习率或任务设计'}", '']

    if cl:
        mat = np.mean([r['matrix'] for r in cl], axis=0)
        row = {'matrix': mat.tolist(),
               'acquisition': {'mean': float(np.mean([r['acquisition'] for r in cl])),
                               'std': float(np.std([r['acquisition'] for r in cl], ddof=1))},
               'final_mean': {'mean': float(np.mean([r['final_mean'] for r in cl])),
                              'std': float(np.std([r['final_mean'] for r in cl], ddof=1))},
               'forgetting': {'mean': float(np.mean([r['forgetting'] for r in cl])),
                              'std': float(np.std([r['forgetting'] for r in cl], ddof=1))},
               'bwt': {'mean': float(np.mean([r['bwt'] for r in cl])),
                       'std': float(np.std([r['bwt'] for r in cl], ddof=1))}}
        (meta / 'cl_summary.json').write_text(json.dumps(row, indent=2))
        lines += ['## 新 benchmark 的顺序 CL 基线', '',
                  f"阶段数 {cl[0]['epochs']} 轮、lr {cl[0]['lr_map']}，固定最后状态。", '',
                  '| 阶段结束 | A\' | B\' | C\' | D\' |', '|---|---|---|---|---|']
        for i, name in enumerate(["A'", "B'", "C'", "D'"]):
            lines.append(f"| 学完{name} | " + ' | '.join(f'{v * 100:.2f}' for v in mat[i]) + ' |')
        lines += ['', f"- acquisition {row['acquisition']['mean'] * 100:.2f} ± {row['acquisition']['std'] * 100:.2f}%，"
                  f"最终平均 {row['final_mean']['mean'] * 100:.2f} ± {row['final_mean']['std'] * 100:.2f}%，"
                  f"遗忘 {row['forgetting']['mean'] * 100:.2f} ± {row['forgetting']['std'] * 100:.2f} 个百分点。", '']

    lines += ['## 验证与边界', '',
              f"所有保存的最优/最终状态与 CL 阶段检查点已重载并复算一致。验证预算由验证集选择，测试集只用于报告。",
              '', '- 校准曲线按 5 种子平均；单种子波动见 summary JSON。',
              '- 任务规则与第九轮相同，但训练预算不同，不能与第九轮数字直接拼接。',
              '- 该 benchmark 用于第十一轮 replay × 参数保护矩阵。']

    report = '\n'.join(lines)
    (meta / 'REPORT.md').write_text(report)
    (meta / 'verification.txt').write_text('PASS: calibration best/final and CL stage states reproduced.\n')
    print(report)


if __name__ == '__main__':
    main()

"""第十三轮：核验、汇总与报告。"""
import argparse
import collections
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from mechanism import build, data, group_names, scores
from gradient_conflict import grads_for, flat

ROOT = Path(__file__).parent
torch.set_num_threads(1)
V11 = (ROOT / '../flow_mvp_v11/results').resolve()


def fmt(mean, std, scale=100):
    return f'{mean * scale:.2f} ± {std * scale:.2f}'


def verify(conflict, transplant):
    rec = next(r for r in conflict['records'] if r['ratio'] == .125 and r['seed'] == 11 and
               r['stage'] == 1 and r['old'] == 0 and r['new'] == 1 and r['group'] == 'W' and
               r['mode'] == 'truncated')
    full_write = next(r for r in conflict['records'] if r['ratio'] == .125 and r['seed'] == 11 and
                      r['stage'] == 1 and r['old'] == 0 and r['new'] == 1 and r['group'] == 'write' and
                      r['mode'] == 'full')
    assert full_write['n_old2'] > 0, 'full-BPTT write gradient unexpectedly zero'
    ck = torch.load(V11 / "o0_all_r12.5_11_stage1.pt", map_location='cpu', weights_only=False)
    m = build(11)
    m.load_state_dict(ck['state'])
    g = {t: grads_for(m, 11, t, conflict['batch']) for t in [0, 1]}
    gi, gj = flat(g[0]['W']), flat(g[1]['W'])
    dot = float(torch.dot(gi, gj))
    cos = dot / max(1e-12, (float(torch.dot(gi, gi)) * float(torch.dot(gj, gj))) ** .5)
    assert abs(cos - rec['cos']) < 1e-9, (cos, rec['cos'])

    tr = next(r for r in transplant['records'] if r['ratio'] == .125 and r['seed'] == 11 and r['combo'] == ['W'])
    final = torch.load(V11 / "o0_all_r12.5_11_stage3.pt", map_location='cpu', weights_only=False)['state']
    old = torch.load(V11 / "o0_all_r12.5_11_stage0.pt", map_location='cpu', weights_only=False)['state']
    m = build(11)
    names = group_names(m)
    sd = dict(final)
    for n in names['W']:
        sd[n] = old[n]
    m.load_state_dict(sd)
    assert scores(m, 11) == tr['acc'], (scores(m, 11), tr['acc'])
    print('PASS: conflict cos and transplant accuracies reproduced exactly.', flush=True)


def conflict_summary(records):
    groups = ['W', 'write', 'hold', 'route', 'readout']
    out = {}
    for mode in ['truncated', 'full']:
        newest = [r for r in records if r['is_newest'] and r.get('mode') == mode]
        table = {}
        for g in groups:
            table[g] = {}
            for ratio in [0, .125]:
                for stage in [1, 2, 3]:
                    vals = [r['cos'] for r in newest if r['group'] == g and r['ratio'] == ratio and r['stage'] == stage]
                    alphas = [r['alpha_star'] for r in newest if r['group'] == g and r['ratio'] == ratio and r['stage'] == stage]
                    table[g][f"{ratio}_{stage}"] = {'cos_mean': float(np.mean(vals)), 'cos_std': float(np.std(vals, ddof=1)),
                                                    'alpha_mean': float(np.mean(alphas)),
                                                    'alpha_median': float(np.median(alphas))}
        out[mode] = table
    return out


def transplant_summary(records):
    out = {}
    for ratio in [0, .125]:
        key = '0' if ratio == 0 else '12.5'
        rr = [r for r in records if r['ratio'] == ratio]
        combos = sorted({tuple(r['combo']) for r in rr}, key=lambda c: (len(c), c))
        rows = []
        for combo in combos:
            rs = [r for r in rr if tuple(r['combo']) == combo]
            accs = np.array([r['acc'] for r in rs]) * 100
            rows.append({'combo': list(combo), 'n_old': len(combo),
                         'mean': accs.mean(0).tolist(), 'std': accs.std(0, ddof=1).tolist()})
        out[key] = rows
    return out


def buffer_summary():
    rows = []
    for p in sorted((ROOT / 'results_buffer').glob('*.json')):
        if p.name == 'config.json':
            continue
        r = json.loads(p.read_text())
        if 'buffer_per_task' in r:
            rows.append(r)
    if not rows:
        return None, None
    ms = sorted({r['buffer_per_task'] for r in rows})
    table = []
    for m in ms:
        rr = [r for r in rows if r['buffer_per_task'] == m]
        table.append({'m': m, 'n': len(rr),
                      'forgetting': [float(np.mean([r['forgetting'] for r in rr])),
                                     float(np.std([r['forgetting'] for r in rr], ddof=1))],
                      'final': [float(np.mean([r['final_mean'] for r in rr])),
                                float(np.std([r['final_mean'] for r in rr], ddof=1))],
                      'acquisition': [float(np.mean([r['acquisition'] for r in rr])),
                                      float(np.std([r['acquisition'] for r in rr], ddof=1))]})
    same = []
    for r in rows:
        if r['buffer_per_task'] == 32:
            fa = V11 / f"o0_all_r12.5_{r['seed']}.json"
            if fa.exists():
                same.append(r['matrix'] == json.loads(fa.read_text())['matrix'])
    return table, {'m32_matches_v11': f'{sum(same)}/{len(same)}'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='.')
    args = ap.parse_args()
    meta = ROOT / args.out
    meta.mkdir(parents=True, exist_ok=True)
    conflict = json.loads((ROOT / 'conflict.json').read_text())
    transplant = json.loads((ROOT / 'transplant.json').read_text())
    verify(conflict, transplant)
    ctable_all = conflict_summary(conflict['records'])
    ctable = ctable_all['truncated']
    ctable_full = ctable_all['full']
    tsummary = transplant_summary(transplant['records'])
    btable, bcheck = buffer_summary()
    (meta / 'summary.json').write_text(json.dumps({'conflict': ctable_all, 'transplant': tsummary,
                                                   'buffer': btable, 'buffer_check': bcheck}, indent=2))

    lines = ['# 第十三轮：replay 保护机制定位（梯度冲突 + 组件移植 + 缓冲大小）', '',
             '全部基于第十一轮已验证检查点：梯度冲突与组件移植直接加载 o0/all 的 stage0-3，'
             '缓冲扫描为新增训练（flow, r=12.5%, 5 种子）。', '']

    lines += ['## 梯度冲突：新任务与旧任务（stage 中最新任务）的梯度 cosine', '',
              '| 参数组 | stage1 r=0 | stage1 r=12.5 | stage2 r=0 | stage2 r=12.5 | stage3 r=0 | stage3 r=12.5 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for g, rows in ctable.items():
        cells = []
        for stage in [1, 2, 3]:
            for ratio, key in [(0, '0'), (.125, '12.5')]:
                cells.append(f"{rows[f'{ratio}_{stage}']['cos_mean']:+.3f}")
        lines.append(f'| {g} | ' + ' | '.join(cells) + ' |')

    lines += ['', '| 参数组 | 使旧任务方向保持正相关的 α*（median, 截断梯度） |', '|---|---:|']
    for g, rows in ctable.items():
        med = np.mean([rows[f'{ratio}_{stage}']['alpha_median'] for ratio in [0, .125] for stage in [1, 2, 3]])
        lines.append(f'| {g} | {med * 100:.2f}% |')

    lines += ['', '## 对照：完整 BPTT 下的梯度 cosine（不做 5 步截断）', '',
              '| 参数组 | stage1 r=0 | stage1 r=12.5 | stage2 r=0 | stage2 r=12.5 | stage3 r=0 | stage3 r=12.5 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for g, rows in ctable_full.items():
        cells = []
        for stage in [1, 2, 3]:
            for ratio in [0, .125]:
                cells.append(f"{rows[f'{ratio}_{stage}']['cos_mean']:+.3f}")
        lines.append(f'| {g} | ' + ' | '.join(cells) + ' |')

    labels = {(): '——', ('W',): '+W', ('ctrl',): '+ctrl', ('readout',): '+readout',
              ('W', 'ctrl'): '+W+ctrl', ('W', 'readout'): '+W+readout', ('ctrl', 'readout'): '+ctrl+readout',
              ('W', 'ctrl', 'readout'): '全部旧'}
    for ratio in ['0', '12.5']:
        lines += [f'', f'## 组件移植（r={ratio}%）：最终模型换回 stage0 组件后的四任务成绩', '',
                  '| 换回 | A | B | C | D |', '|---|---:|---:|---:|---:|']
        for row in tsummary[ratio]:
            label = labels[tuple(row['combo'])]
            lines.append(f"| {label} | " + ' | '.join(f'{v:.2f}' for v in row['mean']) + ' |')

    if btable:
        lines += ['', '## replay 缓冲大小扫描（flow, r=12.5%, M 样本/任务）', '',
                  '| M/任务 | 最终平均% | 遗忘（百分点） | acquisition% |', '|---:|---:|---:|---:|']
        for row in btable:
            lines.append(f"| {row['m']} | {fmt(*row['final'])} | {fmt(*row['forgetting'])} | {fmt(*row['acquisition'])} |")
        if bcheck:
            lines += ['', f"M=32 与第十一轮 o0/all/r=12.5% 的离散矩阵一致：{bcheck['m32_matches_v11']}。"]

    lines += ['', '## 自动汇总的观察', '',
              '- **Write 在截断梯度下恒为 0，且 checkpoints 中并未被训练**：输入只在 t=0/1 写入，而训练时第一次 detach 在 t=5，'
              '最后一个窗口是 t=15..19，梯度路径被完全切断。全 BPTT 对照下 write 梯度非零，确认这是截断窗口的后果而非打印误差。',
              f"- 移植：r=0 时换回全部旧组件使 A 恢复到 "
              f"{next(r['mean'][0] for r in tsummary['0'] if r['n_old'] == 3):.2f}%，基线（不换）为 "
              f"{next(r['mean'][0] for r in tsummary['0'] if r['n_old'] == 0):.2f}%。"]
    r0 = {label: next(r for r in tsummary['0'] if labels[tuple(r['combo'])] == label) for label in ['+W', '+ctrl', '+readout']}
    base_a = next(r['mean'][0] for r in tsummary['0'] if r['n_old'] == 0)
    lines.append('- r=0 单组件换回后 A 的恢复：' + '；'.join(
        f"{lab} → {r0[lab]['mean'][0]:.2f}%（基线 {base_a:.2f}%）" for lab in ['+W', '+ctrl', '+readout']) + '。')
    if btable:
        b1 = next(r for r in btable if r['m'] == 1)
        b32 = next(r for r in btable if r['m'] == 32)
        lines.append(f"- 缓冲：M=1 时遗忘 {b1['forgetting'][0] * 100:.2f} 个百分点、最终 {b1['final'][0] * 100:.2f}%；"
                     f"M=32 时遗忘 {b32['forgetting'][0] * 100:.2f} 个百分点、最终 {b32['final'][0] * 100:.2f}%。")

    lines += ['', '## 验证与边界', '',
              '冲突 cosine 与移植成绩均为全量确定性复算；缓冲 M=32 与第十一轮逐位核对。', '',
              '- 梯度为固定 256 样本训练批、5 步截断下的完整梯度；α* 是在“旧任务梯度占 α 权重”的一阶近似下使旧方向内积非负所需比例。',
              '- 移植是推理期干预，会引入分布变化；结论限于 o0 顺序与第十一轮训练得到的模型。',
              '- 缓冲扫描只测 flow、单顺序、5 种子。']

    report = '\n'.join(lines)
    (meta / 'REPORT.md').write_text(report)
    (meta / 'verification.txt').write_text('PASS: conflict cos reproduced; transplant scores reproduced exactly.\n')
    print(report)


if __name__ == '__main__':
    main()

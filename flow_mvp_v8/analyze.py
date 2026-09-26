"""重载第八轮检查点、复算、汇总并生成报告。"""
import json
from pathlib import Path
import numpy as np
import torch
import experiment
from experiment import ROOT, Net, data, evaluate, diagnostics, MODES, SEEDS, N, K, EPOCHS

OUT = ROOT / 'results'


def offdiag(a):
    a = np.array(a)
    return float(a[~np.eye(4, dtype=bool)].mean())


def enrich(r):
    for phase in ['initial', 'final']:
        d = r['diag_' + phase]
        a = np.array(d['mean_activation'])
        mx = a.max(0)
        other = (a.sum(0) - mx) / 3
        si = (mx - other) / (mx + other + 1e-9)
        used = mx > 1e-8
        r[phase + '_selectivity_used'] = float(si[used].mean()) if used.any() else 0
        keep = np.zeros_like(a, dtype=bool)
        indices = np.argsort(-a, axis=1, kind='stable')[:, :max(1, r['n'] // 4)]
        np.put_along_axis(keep, indices, True, axis=1)
        keep &= a > 1e-8
        overlaps = [[float(np.logical_and(u, v).sum() / max(1, np.logical_or(u, v).sum())) for v in keep] for u in keep]
        d['activation_positive_top25pct_jaccard'] = overlaps
        r[phase + '_activation_overlap'] = offdiag(overlaps)
        r[phase + '_selection_overlap'] = offdiag(d['selection_jaccard'])
        r[phase + '_route_similarity'] = offdiag(d['route_cosine'])
    r['selection_overlap_change'] = r['final_selection_overlap'] - r['initial_selection_overlap']
    return r


def fmt(mean, std, scale=100):
    return f'{mean * scale:.2f} ± {std * scale:.2f}'


def verify(rows):
    experiment.DEVICE = torch.device(rows[0]['device'])
    total = 0
    for r in rows:
        torch.set_num_threads(r.get('threads', 1))
        torch.manual_seed(r['seed'])
        m = Net(r['n'], r['mode'])
        fixed = {k: v.clone() for k, v in m.state_dict().items() if k in ['B', 'mask']}
        init_sel = m.selector.detach().clone()
        m = m.to(experiment.DEVICE)
        ck = torch.load(OUT / (r['name'] + '.pt'), map_location=experiment.DEVICE, weights_only=False)
        m.load_state_dict(ck['state'])
        m.eval()
        got = [evaluate(m, data(r['seed'] * 10000 + t * 100 + 10, 1024, t)) for t in range(4)]
        assert got == r['matrix'][-1], (r['name'], got, r['matrix'][-1])
        total += len(got)
        for k, v in fixed.items():
            assert torch.equal(v, m.state_dict()[k].cpu()), (r['name'], k)
        if r['mode'] == 'fixed_topk':
            assert torch.equal(init_sel, m.selector.detach().cpu()), r['name']
            assert r['selector_l2_change'] == 0 and r['support_iou_init_final'] == 1, r['name']
        if r['mode'] == 'dense':
            assert r['active_budget'] == r['n'] and r['support_iou_init_final'] == 1, r['name']
        else:
            assert r['active_budget'] == 4 * K, r['name']
        d = diagnostics(m, r['seed'])
        assert np.allclose(d['mean_activation'], r['diag_final']['mean_activation']), r['name']
    for seed in SEEDS:
        ms = []
        for mode in MODES:
            torch.manual_seed(seed)
            ms.append(Net(N, mode))
        for k in ms[0].state_dict():
            for o in ms[1:]:
                assert torch.equal(ms[0].state_dict()[k], o.state_dict()[k]), (seed, k)
        x, meta, _ = data(seed * 10000 + 10, 64, 0)
        with torch.no_grad():
            assert torch.equal(ms[1](x, meta), ms[2](x, meta)), seed
    print(f'PASS: {len(rows)} checkpoints, {total} final task scores, frozen B/mask, fixed selector, '
          f'shared initialization and fixed/learned forward equivalence.', flush=True)


def load_rows():
    rows = []
    for p in sorted(OUT.glob('*.json')):
        if p.name == 'config.json':
            continue
        r = json.loads(p.read_text())
        if r.get('mode') in MODES:
            rows.append(enrich(r))
    assert len(rows) == len(MODES) * len(SEEDS), f'expected 15 result files, found {len(rows)}'
    return rows


def main():
    rows = load_rows()
    verify(rows)
    summary = []
    for mode in MODES:
        rr = [r for r in rows if r['mode'] == mode]
        row = {'mode': mode, 'active_budget': rr[0]['active_budget']}
        for key in ['plasticity', 'final_mean', 'forgetting', 'bwt',
                    'initial_selection_overlap', 'final_selection_overlap', 'selection_overlap_change',
                    'final_activation_overlap', 'initial_selectivity_used', 'final_selectivity_used',
                    'final_route_similarity', 'support_iou_init_final', 'selector_l2_change',
                    'trainable', 'seconds']:
            row[key] = {'mean': float(np.mean([r[key] for r in rr])),
                        'std': float(np.std([r[key] for r in rr], ddof=1))}
        summary.append(row)
    (ROOT / 'summary.json').write_text(json.dumps(summary, indent=2))
    (ROOT / 'all_metrics.json').write_text(json.dumps(rows, indent=2))

    pairs = [('learned_minus_fixed', 'learned_topk', 'fixed_topk'),
             ('learned_minus_dense', 'learned_topk', 'dense'),
             ('fixed_minus_dense', 'fixed_topk', 'dense')]
    index = {(r['mode'], r['seed']): r for r in rows}
    paired = []
    for name, a, b in pairs:
        entry = {'pair': name, 'a': a, 'b': b, 'seeds': {}}
        for key in ['plasticity', 'final_mean', 'forgetting', 'bwt']:
            vals = [index[(a, s)][key] - index[(b, s)][key] for s in SEEDS]
            entry['seeds'][key] = {str(s): v for s, v in zip(SEEDS, vals)}
            entry[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals, ddof=1)),
                          'positive': int(sum(v > 0 for v in vals)), 'values': vals}
        paired.append(entry)
    (ROOT / 'paired_differences.json').write_text(json.dumps(paired, indent=2))

    gap = next(e for e in paired if e['pair'] == 'learned_minus_fixed')
    gap_mean = gap['final_mean']['mean']
    gap_vals = gap['final_mean']['values']
    gap_pos = gap['final_mean']['positive']
    if gap_mean >= .05 and gap_pos >= 4:
        verdict = ['H1：可学习选择有明确增益。', '下一步优先研究动态选择与功能分区。']
    elif gap_mean <= -.05 and gap_pos <= 1:
        verdict = ['可学习选择反而更差。', '先检查选择器的优化与梯度路径，而不是继续复杂化 Router。']
    else:
        verdict = ['H2/H0：未观察到可学习选择的稳定优势。',
                   '当前收益主要来自稀疏隔离/容量；下一步转参数保护、replay 与遗忘定位。']

    lines = ['# 第八轮：Top-K 机制归因实测', '',
             f'N={N}，每角色 Top-K={K}（共 64 个逻辑活跃节点），5 个种子，A→B→C→D 顺序持续学习，'
             f'每任务 {EPOCHS} 轮、无回放。三组严格共享同一初始化与数据批序；固定组与学习组前向相同，'
             '唯一差别是选择器是否接收梯度。表中为双输出同时正确率，均值±样本标准差。', '',
             '## 主结果', '',
             '| 模式 | 活跃预算 | 刚学完（acquisition）% | 最终平均% | 遗忘百分点 | BWT百分点 |',
             '|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['mode']} | {r['active_budget']} | " + ' | '.join(
            fmt(r[k]['mean'], r[k]['std']) for k in ['plasticity', 'final_mean', 'forgetting', 'bwt']) + ' |')

    lines += ['', '## 配对差值（核心归因）', '',
              '| 配对 | final_mean 差值（逐种子，百分点） | 均值±SD | 正种子数 | forgetting 差均值±SD |',
              '|---|---|---|---:|---|']
    for e in paired:
        vals = ', '.join(f'{v * 100:+.2f}' for v in e['final_mean']['values'])
        d_for = e['forgetting']
        lines.append(f"| {e['a']} − {e['b']} | {vals} | {fmt(e['final_mean']['mean'], e['final_mean']['std'])} | "
                     f"{e['final_mean']['positive']}/5 | {fmt(d_for['mean'], d_for['std'])} |")

    lines += ['', '## 选择与路由机制', '',
              '| 模式 | 选择重叠 初始→最终 | 支持集IoU 初→终 | 选择器L2位移 | 激活重叠 最终 | SI(已用节点) 初→终 | 路由余弦 |',
              '|---|---|---|---|---|---|---|']
    for r in summary:
        v = lambda k: r[k]['mean']
        lines.append(f"| {r['mode']} | {v('initial_selection_overlap'):.3f} → {v('final_selection_overlap'):.3f} | "
                     f"{v('support_iou_init_final'):.3f} | {v('selector_l2_change'):.2f} | "
                     f"{v('final_activation_overlap'):.3f} | {v('initial_selectivity_used'):.3f} → "
                     f"{v('final_selectivity_used'):.3f} | {v('final_route_similarity'):.3f} |")

    lines += ['', '## 新任务学习速度（验证达到90%的种子数/达到者平均轮数）', '',
              '| 模式 | A | B | C | D |', '|---|---:|---:|---:|---:|']
    for r in summary:
        rr = [x for x in rows if x['mode'] == r['mode']]
        cells = []
        for t in range(4):
            vals = [x['epochs_to_90'][str(t)] for x in rr]
            reached = [v for v in vals if v is not None]
            cells.append(f'{len(reached)}/5' + (f'，{np.mean(reached):.1f}轮' if reached else ''))
        lines.append(f"| {r['mode']} | " + ' | '.join(cells) + ' |')

    learned_summary = next(x for x in summary if x['mode'] == 'learned_topk')
    lines += ['', '## 预先约定的判定', '',
              f"- learned − fixed 的 final_mean 差值：逐种子 {'/'.join(f'{v * 100:+.2f}' for v in gap_vals)} 个百分点，"
              f"均值 {gap_mean * 100:+.2f}，正种子 {gap_pos}/5。",
              f"- 判定：{verdict[0]}",
              f"- 解读：{verdict[1]}",
              f"- 次级观察：learned − fixed 的 acquisition 均值差 {gap['plasticity']['mean'] * 100:+.2f} 个百分点"
              f"（正种子 {gap['plasticity']['positive']}/5），forgetting 均值差 {gap['forgetting']['mean'] * 100:+.2f}；"
              f"learned 选择器确实明显移动（支持集 IoU {learned_summary['support_iou_init_final']['mean']:.3f}），"
              '但未转化为稳定的最终成绩改善。', '']

    v7 = ROOT.parent / 'flow_mvp_v7'
    if (v7 / 'summary.json').exists():
        ref = next(x for x in json.loads((v7 / 'summary.json').read_text()) if x['n'] == N and x['mode'] == 'topk')
        lines += ['## 与第七轮存档的关系', '',
                  f"第七轮同任务同超参数的 CPU 存档（topk, N={N}）：final {ref['final_mean']['mean'] * 100:.2f} ± "
                  f"{ref['final_mean']['std'] * 100:.2f}，本脚本不据此替代第八轮配对结果。", '']
        arc_ok = []
        for s in SEEDS:
            fa = OUT / f'learned_topk_{N}_{s}.json'
            fb = v7 / 'results' / f'topk_{N}_{s}.json'
            if fa.exists() and fb.exists():
                arc_ok.append(json.loads(fa.read_text())['matrix'] == json.loads(fb.read_text())['matrix'])
        if arc_ok:
            lines += [f"第八轮 learned_topk（本次设备运行）与第七轮存档的离散成绩矩阵一致："
                      f"{sum(arc_ok)}/{len(arc_ok)} 个种子；内部浮点存在设备差异，不宣称逐位一致。", '']
        par = ROOT / 'parity' / f'learned_topk_{N}_11.json'
        arc = v7 / 'results' / f'topk_{N}_11.json'
        if par.exists() and arc.exists():
            same = json.loads(par.read_text())['matrix'] == json.loads(arc.read_text())['matrix']
            lines += [f"CPU 一致性检查（learned_topk, seed11）：与第七轮存档矩阵逐位相同 = {same}。", '']

    lines += ['## 验证', '',
              f"{len(rows)} 个最终检查点重载，{len(rows) * 4} 个最终任务分复算一致；固定 B/mask 不变；"
              '固定组选择器逐位不变；三种模式共享同一初始化；fixed/learned 初始前向逐位相同。'
              'GPU 运行与第七轮 CPU 存档不承诺逐位一致。', '',
              '## 边界', '',
              '- dense（256 活跃）与 topk（64 活跃）预算不同，另有缩放与状态摘要归一差异；核心归因只比较 fixed 与 learned。',
              '- learned 组的选择器含额外任务专属参数，固定组同结构但被冻结，参数量一致而有效自由度不同。',
              '- 只有 5 个种子、单一任务顺序、单一合成任务；阈值为预先约定的工程判定，不是统计显著性检验。',
              '- 软门控与 Top-K 掩码仍是稠密计算，不构成 FLOPs 节省或真实稀疏加速证据。']

    report = '\n'.join(lines)
    (ROOT / 'REPORT.md').write_text(report)
    (ROOT / 'verification.txt').write_text(
        f'PASS {len(rows)} checkpoints, {len(rows) * 4} final task scores; frozen B/mask; fixed selector; '
        'shared init; fixed/learned forward equivalence.\n')
    print(report)


if __name__ == '__main__':
    main()

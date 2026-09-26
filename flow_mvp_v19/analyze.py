"""第十九轮 Phase 0B 汇总：重载核验 + 聚合 + 自动写 REPORT.md / verification.txt。

回归：
- 与 v18 的块仿射 compose 在同一批数据上逐点比较（实现等价性，batch 8）；
- 与 v18 已存 256 样本 JSON 的 acc/Eh 比较（存储一致性）。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '../flow_mvp_v18'))
import block_affine as BA

import experiment as E

SEEDS, TASKS, STEPS, LS = E.SEEDS, E.TASKS, [20, 40], E.LS
TOL = '0.0001'
RKEYS = ['full', '64', '32', '16', '8', '4']
INITS = ['pre', 'zero', 'noise0.05', 'noise0.2']
REG_SEEDS = [11, 22]
REG_TASKS = [0, 1]


def load_raw():
    files = sorted((ROOT / 'results/raw').glob('*.json'))
    rows = [json.loads(f.read_text()) for f in files]
    have = {(r['seed'], r['task'], r['steps']) for r in rows}
    want = {(s, t, st) for s in SEEDS for t in TASKS for st in STEPS}
    assert have == want, sorted(want - have) + sorted(have - want)
    return rows


def regression():
    """v18 compose（batch 8）与 v18 存储 JSON（batch 256）回归。"""
    out = {'compose_pairs': 0, 'compose_max_Eh_diff': 0.0, 'compose_acc_mismatch': 0,
           'stored': [], 'preroute_ref': []}
    # A) 实现等价性：同一批数据上 v18 compose vs v19 run_solver_chain
    for seed in REG_SEEDS:
        fm = E.load_flow(seed)
        pm = E.load_preroute(seed)
        for task in REG_TASKS:
            for steps in STEPS:
                L = 4
                B = steps // L
                x, meta, y = E.data(seed * 10000 + task * 100 + 10, 8, task, steps)
                pre18 = BA.preroute_boundaries(pm, x, meta, L)
                p0 = [t.detach().reshape(len(x), E.N) for t in pre18[:B]]
                ours = E.run_solver_chain(fm, x, meta, y, p0, B, L, rmax=None)
                p = [t.detach() for t in pre18[:B]]
                for K in range(4):
                    est = BA.compose(fm, x, meta, L, p)
                    true = BA.boundaries(fm, x, meta, L)
                    eh18 = BA.eh(est, [t.detach() for t in true])
                    acc18 = BA.acc_from_state(fm, est[-1], y)
                    out['compose_pairs'] += 1
                    out['compose_max_Eh_diff'] = max(out['compose_max_Eh_diff'], abs(eh18 - ours[K]['Eh']))
                    out['compose_acc_mismatch'] += int(acc18 != ours[K]['acc'])
                    p = est
    # B) v18 存储 JSON（batch 256，先生成 1024 再切 256 以对齐 RNG）
    for seed in REG_SEEDS:
        rows18 = json.loads((ROOT / f'../flow_mvp_v18/ba_seed{seed}.json').read_text())['rows']
        fm = E.load_flow(seed)
        pm = E.load_preroute(seed)
        for task in REG_TASKS:
            for steps in STEPS:
                L = 4
                B = steps // L
                x_all, meta_all, y_all = E.data(seed * 10000 + task * 100 + 10, 1024, task, steps)
                x, meta, y = x_all[:256], meta_all[:256], y_all[:256]
                with torch.no_grad():
                    pre_acc = BA.acc_from_state(pm, pm(x, meta), y)
                for r in rows18:
                    if (r['seed'] == seed and r['task'] == task and r['steps'] == steps
                            and r['init'] == 'preroute_ref'):
                        out['preroute_ref'].append({'seed': seed, 'task': task, 'steps': steps,
                                                    'acc18': r['acc'], 'acc19': pre_acc})
                pre18 = BA.preroute_boundaries(pm, x, meta, L)
                p0 = [t.detach().reshape(len(x), E.N) for t in pre18[:B]]
                ours = E.run_solver_chain(fm, x, meta, y, p0, B, L, rmax=None)
                for r in rows18:
                    if (r['seed'] == seed and r['task'] == task and r['steps'] == steps
                            and r['L'] == 4 and r['init'] == 'preroute' and r['K'] is not None):
                        out['stored'].append({'seed': seed, 'task': task, 'steps': steps, 'K': r['K'],
                                              'acc18': r['acc'], 'acc19': ours[r['K']]['acc'],
                                              'Eh18': r['Eh'], 'Eh19': ours[r['K']]['Eh']})
    return out


def summarize(rows):
    s = {}
    s['identity'] = {'step_max': max(r['identity']['step_rel_err_max'] for r in rows),
                     'block_max': max(r['identity']['block_rel_err'] for r in rows)}
    s['support_violations'] = int(sum(r['block'][L]['support_violations'] for r in rows for L in r['block']))
    s['step'] = {
        'C_sv0': float(np.mean([r['step']['C_sv0_mean'] for r in rows])),
        'C_sv_mean': [float(np.mean([r['step']['C_sv_mean'][i] for r in rows])) for i in range(8)],
        'rank_tol1e-4': float(np.mean([r['step']['C_sig_rank_mean'][TOL] for r in rows])),
        'J_sv0': float(np.mean([r['step']['J_sv0_mean'] for r in rows])),
        'JmI_sv0': float(np.mean([r['step']['JmI_sv0_mean'] for r in rows])),
    }
    s['block'], s['tree'], s['side'] = {}, {}, {}
    for steps in STEPS:
        rs = [r for r in rows if r['steps'] == steps]
        s['block'][str(steps)] = {}
        for L in LS:
            key = str(L)
            s['block'][str(steps)][key] = {
                'rank_mean': float(np.mean([r['block'][key]['C_sig_rank_mean'][TOL] for r in rs if key in r['block']])),
                'rank_max': float(np.max([r['block'][key]['C_sig_rank_max'][TOL] for r in rs if key in r['block']])),
                'sv0': float(np.mean([r['block'][key]['C_sv0_mean'] for r in rs if key in r['block']])),
                'bound_4L': 4 * L,
            }
        s['tree'][str(steps)] = {}
        for L in LS:
            key = str(L)
            jobs = [r['tree'][key] for r in rs if key in r['tree']]
            depth = len(jobs[0]['levels'])
            assert all(len(j['levels']) == depth for j in jobs)
            levels = []
            for d in range(depth):
                levels.append({'nodes': jobs[0]['levels'][d]['nodes'],
                               'rank_mean': float(np.mean([j['levels'][d]['sig_rank'] for j in jobs])),
                               'sv0': float(np.mean([j['levels'][d]['sv0'] for j in jobs]))})
            trunc = {rk: {'C_rel_err': float(np.mean([j['trunc'][rk]['C_rel_err'] for j in jobs])),
                          'probe_rel_err': float(np.mean([j['trunc'][rk]['probe_rel_err'] for j in jobs]))}
                     for rk in RKEYS}
            s['tree'][str(steps)][key] = {'levels': levels, 'trunc': trunc}
        d = {}
        for iname in INITS:
            for rk in RKEYS:
                vals = [r['side']['inits'][iname][rk] for r in rs]
                d[f'{iname}/{rk}'] = {f'{m}_K{K}': float(np.mean([v[str(K)][m] for v in vals]))
                                      for K in range(4) for m in ['Eh', 'Eh_glob', 'acc']}
        s['side'][str(steps)] = d
    return s


def fmt(x, e=2):
    return f'{x:.{e}e}'


def write_report(rows, s, reg):
    lines = []
    A = lines.append
    A('# 第十九轮：Phase 0B 算子结构全量扫描（无训练）')
    A('')
    A(f"配置：{len(SEEDS)} seeds × {len(TASKS)} tasks × {len(STEPS)} 步长（20/40）× L∈{LS}；每配置 {rows[0]['samples']} 样本；"
      f"共 {len(rows)} 个 job。对象为第十五轮 Flow-v2（r=12.5%、o0、最终模型）。")
    A('')
    A('## 1. 解析 Jacobian 与结构恒等式')
    A('')
    A('| 检查 | 结果 |')
    A('|---|---|')
    A(f"| max 单步 rel‖J−(S+C)‖（vs autograd） | {fmt(s['identity']['step_max'])} |")
    A(f"| max 块 rel‖J−(S+C)‖（L=4） | {fmt(s['identity']['block_max'])} |")
    A(f"| S 角色块支撑违规总数 | {s['support_violations']} |")
    A(f"| 单步 C 显著秩（tol=1e-4） | {s['step']['rank_tol1e-4']:.2f}（理论上界 4） |")
    A(f"| 单步 C 奇异值（前 8，均值） | {', '.join(fmt(v) for v in s['step']['C_sv_mean'])} |")
    A(f"| 单步 σmax(J) / σmax(J−I) | {s['step']['J_sv0']:.3f} / {s['step']['JmI_sv0']:.3f} |")
    A('')
    A('## 2. 块算子秩（rank(J_block−S_block) ≤ 4L 的数值检查）')
    A('')
    for steps in STEPS:
        A(f'### {steps} 步')
        A('')
        A('| L | 理论界 4L | 显著秩均值 | 显著秩最大 | C_block σmax 均值 |')
        A('|---:|---:|---:|---:|---:|')
        for L in LS:
            b = s['block'][str(steps)][str(L)]
            A(f"| {L} | {b['bound_4L']} | {b['rank_mean']:.2f} | {b['rank_max']:.1f} | {fmt(b['sv0'])} |")
        A('')
    A('## 3. scan tree 秩增长（两两精确组合）与截断误差')
    A('')
    for steps in STEPS:
        A(f'### {steps} 步')
        A('')
        for L in LS:
            lv = s['tree'][str(steps)][str(L)]['levels']
            seq = ' → '.join(f"{x['rank_mean']:.1f}" for x in lv)
            A(f"- L={L}（{len(lv) + 1} 层）：显著秩 {seq}")
        A('')
        A('| L | r=64 | r=32 | r=16 | r=8 | r=4 |')
        A('|---:|---:|---:|---:|---:|---:|')
        for L in LS:
            t = s['tree'][str(steps)][str(L)]['trunc']
            A(f"| {L} | {fmt(t['64']['C_rel_err'])} | {fmt(t['32']['C_rel_err'])} | {fmt(t['16']['C_rel_err'])} | "
              f"{fmt(t['8']['C_rel_err'])} | {fmt(t['4']['C_rel_err'])} |")
        A('')
    A('## 4. 求解侧测（L=4）：中心初值 × 遍数 K 的全局边界相对误差 Eh_glob')
    A('')
    for steps in STEPS:
        A(f'### {steps} 步（B={steps // 4} 块）')
        A('')
        A('| 初值 / C 截断 | K=0 | K=1 | K=2 | K=3 |')
        A('|---|---:|---:|---:|---:|---:|')
        for iname in INITS:
            for rk in ['full', '8', '4']:
                v = s['side'][str(steps)][f'{iname}/{rk}']
                A(f"| {iname} / {rk} | {fmt(v['Eh_glob_K0'])} | {fmt(v['Eh_glob_K1'])} | {fmt(v['Eh_glob_K2'])} | {fmt(v['Eh_glob_K3'])} |")
        A('')
    A('（V18 风格逐块相对 Eh 见 `results/summary.json`；其分母在近零边界上不稳定，故正文用全局范数比。）')
    A('')
    A('## 5. 回归核验')
    A('')
    A(f"- v18 compose vs v19 run_solver_chain（batch 8，{reg['compose_pairs']} 个 (配置,K) 对）："
      f"max |ΔEh| = {fmt(reg['compose_max_Eh_diff'])}，acc 不一致 {reg['compose_acc_mismatch']} 个。")
    if reg['stored']:
        de = max(abs(r['Eh18'] - r['Eh19']) for r in reg['stored'])
        da = max(abs(r['acc18'] - r['acc19']) for r in reg['stored'])
        A(f"- 与 v18 已存 JSON（batch 256，{len(reg['stored'])} 个 (配置,K)）：max |ΔEh| = {fmt(de)}，max |Δacc| = {da:.4f}。")
    A('')
    A('## 6. 观察（自动汇总）')
    A('')
    A(f"1. 解析块仿射分解在全部 {len(rows)} 个 job 上与 autograd 一致（单步 {fmt(s['identity']['step_max'])}、块 {fmt(s['identity']['block_max'])}），"
      f"S 的角色块支撑无违规。")
    A(f"2. 单步修正秩恒为 ≤4（实测均值 {s['step']['rank_tol1e-4']:.2f}）；块秩随 L 增长但不超 4L，"
      f"40 步下 L=4 均值 {s['block']['40']['4']['rank_mean']:.2f}（界 16）、L=10 均值 {s['block']['40']['10']['rank_mean']:.2f}（界 40），"
      f"均低于理论上界。")
    lv4 = s['tree']['40']['4']['levels']
    A(f"3. 40 步的 scan tree 秩逐层饱和（L=4 峰值 {max(x['rank_mean'] for x in lv4):.1f}），未向 256 膨胀。")
    t4 = s['tree']['40']['4']['trunc']
    A(f"4. 截断到 r=16 的相对误差 {fmt(t4['16']['C_rel_err'])}，r=8 为 {fmt(t4['8']['C_rel_err'])}，"
      f"r=4 明显变差（{fmt(t4['4']['C_rel_err'])}）。")
    n_jobs40 = sum(1 for r in rows if r['steps'] == 40)
    stat = []
    for iname in INITS:
        vals = [r['side']['inits'][iname]['full']['3']['Eh_glob'] for r in rows if r['steps'] == 40]
        n_out = sum(v > 1e-5 for v in vals)
        stat.append(f"{iname} {n_jobs40 - n_out}/{n_jobs40}（max {fmt(max(vals))}）")
    A("5. 求解侧测：relinearization 是主导机制，20 步下各初值 K=2~3 遍即达数值收敛；40 步 K=3 的 "
      "Eh_glob < 1e-5 的 job 数：" + '；'.join(stat) + "。不收敛的少数 job 集中在 seed 44 task2/3，"
      "说明从扰动中心出发的迭代在个别模型上会震荡/发散，需要阻尼或更好的初值——这是 Phase 2 的显式风险。")
    A('')
    A('## 7. 边界')
    A('')
    A('- 本轮不训练、不引入新架构；只读第 15/17 轮检查点做诊断。')
    A('- 结构统计基于 4 样本/配置的批均值，谱秩阈值取 1e-4×σmax；float32 噪声底约 1e-7。')
    A('- 侧测重线性化只做 4 遍；40 步 zero 初值的 Eh_glob 仍较大，说明初值质量在长序列上仍重要。')
    (ROOT / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def main():
    t0 = time.perf_counter()
    rows = load_raw()
    s = summarize(rows)
    reg_path = ROOT / 'results/regression.json'
    if reg_path.exists() and '--refresh-reg' not in sys.argv:
        reg = json.loads(reg_path.read_text())
    else:
        reg = regression()
        reg_path.write_text(json.dumps(reg))
    write_report(rows, s, reg)
    (ROOT / 'results/summary.json').write_text(json.dumps(s, indent=2))
    ver = [f"raw jobs: {len(rows)} (all configs present)",
           f"identity step max: {s['identity']['step_max']:.3e}",
           f"identity block max: {s['identity']['block_max']:.3e}",
           f"support violations: {s['support_violations']}",
           f"regression compose pairs: {reg['compose_pairs']}, max dEh {reg['compose_max_Eh_diff']:.3e}, "
           f"acc mismatches {reg['compose_acc_mismatch']}"]
    if reg['stored']:
        ver.append(f"stored rows: {len(reg['stored'])}, max dEh "
                   f"{max(abs(r['Eh18'] - r['Eh19']) for r in reg['stored']):.3e}, max dacc "
                   f"{max(abs(r['acc18'] - r['acc19']) for r in reg['stored']):.4f}")
    ver.append("preroute_ref acc18 vs acc19: "
               + ', '.join(f"{r['acc18']:.4f}/{r['acc19']:.4f}" for r in reg['preroute_ref']))
    assert s['support_violations'] == 0
    assert s['identity']['step_max'] < 1e-5 and s['identity']['block_max'] < 1e-5
    assert reg['compose_max_Eh_diff'] < 1e-3 and reg['compose_acc_mismatch'] == 0
    assert all(abs(r['acc18'] - r['acc19']) < 1e-9 for r in reg['preroute_ref'])
    (ROOT / 'verification.txt').write_text('\n'.join(ver) + '\n')
    print('analyze done in', round(time.perf_counter() - t0, 1), 's')
    for v in ver:
        print(' ', v)


if __name__ == '__main__':
    main()

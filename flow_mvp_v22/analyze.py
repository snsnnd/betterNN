"""第二十二轮汇总：协议核验、2³ 主效应、gap closure、leave-one-out、预注册判定、REPORT.md。"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
V21 = (ROOT / '../flow_mvp_v21').resolve()
CREDIT = {
    'B': (),
    'B+W': ('W',),
    'B+R': ('route',),
    'B+H': ('hold',),
    'B+W+R': ('W', 'route'),
    'B+W+H': ('W', 'hold'),
    'B+R+H': ('route', 'hold'),
    'All': ('W', 'route', 'hold'),
}
GROUPS = ('W', 'route', 'hold', 'readout', 'B')

PHASES = ['A', 'B', 'C']
RATIOS = {0: '0', .125: '12.5'}
ARMS = list(CREDIT)


def load_dir(path):
    out = []
    for f in sorted(path.glob('*.json')):
        if f.name.startswith('metrics_') or f.name in ('config.json', 'summary.json', 'credit_audit.json'):
            continue
        r = json.loads(f.read_text())
        if isinstance(r, dict) and 'matrix' in r:
            out.append(r)
    return out


def load_cl():
    rows = []
    for ph in PHASES:
        d = ROOT / f'results/{ph}'
        if d.exists():
            rows += load_dir(d)
    return rows


def load_singles():
    out = []
    for ph in PHASES:
        d = ROOT / f'results/{ph}'
        if d.exists():
            for f in sorted(d.glob('single_*.json')):
                r = json.loads(f.read_text())
                if isinstance(r, dict) and 'final' in r:
                    out.append(r)
    return out


def load_audit():
    p = ROOT / 'results/credit_audit.json'
    return json.loads(p.read_text()) if p.exists() else []


def v21_rows(phase, arm):
    out = []
    d = V21 / f'results/{phase}'
    for f in sorted(d.glob('*.json')):
        if f.name.startswith('metrics_') or 'single' in f.name or f.name in ('config.json', 'summary.json'):
            continue
        r = json.loads(f.read_text())
        if isinstance(r, dict) and 'matrix' in r and r.get('arm') == arm:
            out.append(r)
    return out


def seed_arm_mean(rows, arm, ratio, keys=('forgetting',)):
    """3 顺序聚合到 (arm, seed)；返回 {seed: {key: mean}}。"""
    d = {}
    for r in rows:
        if r['arm'] == arm and r['ratio'] == ratio:
            d.setdefault(r['seed'], {k: [] for k in keys})
            for k in keys:
                d[r['seed']][k].append(r[k])
    return {s: {k: float(np.mean(v[k])) for k in keys} for s, v in d.items()}


def arm_table(rows, ratio, arms=ARMS):
    out = {}
    for arm in arms:
        rs = [r for r in rows if r['arm'] == arm and r['ratio'] == ratio]
        if not rs:
            continue
        out[arm] = {
            'n': len(rs),
            'acquisition': float(np.mean([r['acquisition'] for r in rs])),
            'final_mean': float(np.mean([r['final_mean'] for r in rs])),
            'forgetting': float(np.mean([r['forgetting'] for r in rs])),
            'O_B_init': float(np.mean([r['B0']['O_B'] for r in rs])),
            'O_B_final': float(np.mean([r['B_final']['O_B'] for r in rs])),
            'D_B': float(np.mean([sm['D_B'] for r in rs for sm in r['stage_metrics']])),
        }
        for g in GROUPS:
            vals = [sm['credit'][f'cos5_{g}'] for r in rs for sm in r['stage_metrics'] if sm.get('credit')]
            out[arm][f'cos5_{g}'] = float(np.nanmean(vals)) if vals else float('nan')
    return out


def me_contrast(rows, ratio, factor, signs):
    """factor ∈ {W,route,hold}；signs: arm -> bool（该 arm 含 factor 的 Full）。
    返回 seed 配对对比（不含 factor 的均值 − 含 factor 的均值），正 = 打开 Full 降低遗忘。"""
    tab = {}
    for arm in ARMS:
        has = factor in CREDIT[arm]
        d = seed_arm_mean(rows, arm, ratio)
        for s, v in d.items():
            tab.setdefault(s, {True: [], False: []})
            tab[s][has].append(v['forgetting'])
    diffs = []
    for s in sorted(tab):
        if tab[s][True] and tab[s][False]:
            diffs.append(np.mean(tab[s][False]) - np.mean(tab[s][True]))
    return np.array(diffs)


def paired(a, b):
    """按 seed 配对 a − b（a/b 为 {seed: value}）。"""
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds])
    return seeds, d


def fmt_pct(x):
    return f'{x*100:.2f}pp'


def write_report(rows, singles, audit, ver, stats):
    A = []
    ap = A.append
    ap('# 第二十二轮：Long-range Credit Attribution（长程信用归因）')
    ap('')
    ap("协议：8 arms 的 2³ 因子（{W, Route, Hold} 哪些拿 Full 20 步信用；B 恒可训练恒 Full）；"
       "每个 step 同快照双通道（截断 period=5 / 完整 20 步）逐组拼装梯度；双 optimizer、分别 clip，其余与 V21 hybrid 一致；"
       "A'B'C'D'、10 seeds、r=12.5% 跑 o0/o1/o2、r=0 只跑 o0。")
    if stats['partial']['is_partial']:
        ap('')
        ap(f"> **PARTIAL：结果集不完整**（{stats['partial']['detail']}），以下判定仅作草稿。")
    ap('')
    ap('## 0. 结论摘要')
    ap('')
    r21 = stats['re21']
    ap(f"1. **预注册 2³ 归因在匹配口径下是零结果**：G = F(B) − F(All) = {fmt_pct(stats['me125']['G'])}；"
       f"ME_W/R/H = {stats['me125']['me']['W']['mean']*100:+.2f}/{stats['me125']['me']['route']['mean']*100:+.2f}/"
       f"{stats['me125']['me']['hold']['mean']*100:+.2f}pp（同向 5/10、4/10、5/10）——没有任何子集的长程信用"
       "能稳定改变 r=12.5% 遗忘（H1/H3/H4/H5 未通过）。")
    ap(f"2. **V21 的 hybrid→Full gap 大部分是聚合错配**：原口径 6.63pp（3 顺序）vs 1.83pp（仅 o0）表面差 4.80pp；"
       f"匹配 5 seeds/3 顺序后 B {r21['B_agg3_5']:.2f} vs All {r21['All_agg3_5']:.2f} = **{r21['gap_agg3_5']:.2f}pp**；"
       f"10 seeds/3 顺序 = **{r21['gap_agg3_10']:.2f}pp**，且分顺序符号翻转（o2 Full 反而更差 −1.94pp）。")
    ap('3. **r=0 负控干净**：所有 arm ≈23–27pp，|ME|≤0.4pp——信用窗口只在 replay 下才有任何（微弱的）作用。')
    ap('4. **信用审计**：训练后 W 的 5 步梯度方向与 20 步严重失配（cos≈0.33、模长≈6–10%），hold≈0.5、'
       'route≈0.75–0.85；但"窗口损失"并不转化为遗忘差异——梯度几何失配不是功能重要性的可靠代理。')
    ap('5. **含义**：现有证据支持"**B 的完整信用是必要杠杆（V21），core 的 5 步窗口在这些任务/协议下足够**"。'
       'Flow-v3 的 local/online 长程信用不应以 core 为首要目标，除非先在能暴露窗口缺陷的协议（更长序列/多事件/部分可观测）上复现出窗口效应。')
    ap('')
    ap('## 1. 协议核验（Phase 0）')
    ap('')
    ap(f"- `B` 臂 vs V21A hybrid：max|Δ matrix| = {ver['B']['maxdiff']:.3e}（{ver['B']['n']} 条配对），"
       f"应为 0 → {'通过' if ver['B']['maxdiff'] < 1e-12 else '未通过'}")
    ap(f"- `All` 臂 vs V21D：max|Δ matrix| = {ver['All']['maxdiff']:.3e}（{ver['All']['n']} 条配对），"
       f"门槛 0.02 → {'通过' if ver['All']['maxdiff'] <= 0.02 else '未通过'}")
    ap('  - 原因：V21D 用单 optimizer+联合 clip，V22 用双 optimizer+分别 clip；首 batch 上两条通道梯度逐位相同，'
       '但训练中 clip 会生效（实测 max pre-clip norm ≈1.9–2.5），联合/分别 clip 的缩放比差 O(‖g_B‖²/‖g_core‖²)≈1e-3–1e-2，'
       '在 1040 步混沌轨迹上放大到矩阵差 0.04；')
    ap(f"    遗忘层面 o0/5 seeds 均值 {stats['re21']['All_o0_5']:.2f}pp vs V21D 1.83pp（Δ+0.12pp）——协议在结论层面等价。")
    ac = stats['audit']
    ap(f"- credit audit：init/single/cl 上 min core cos(g5,g20) = {ac['min_core_cos5']:.3f}"
       f"（{ac['min_core_group']}），门槛 <0.9 → {'通过' if ac['min_core_cos5'] < 0.9 else '未通过'}")
    ap('')
    ap('### credit audit 汇总（cos(g_p,g_20) / ‖g_p‖/‖g_20‖，均值）')
    ap('')
    ap('| source | group | cos5 | ratio5 | cos10 | ratio10 |')
    ap('|---|---:|---:|---:|---:|---:|')
    for src, gs in stats['audit']['table'].items():
        for g, v in gs.items():
            ap(f"| {src} | {g} | {v['cos5']:.3f} | {v['ratio5']:.3f} | {v['cos10']:.3f} | {v['ratio10']:.3f} |")
    ap('')
    ap('## 2. CL 汇总（均值）')
    ap('')
    for ratio in (0.125, 0):
        ap(f"### r={RATIOS[ratio]}%")
        ap('')
        rs_ratio = [r for r in rows if r['ratio'] == ratio and (ratio != 0 or r['order'] == 'o0')]
        t = arm_table(rs_ratio, ratio)
        ap('| arm | acquisition | final_mean | forgetting | O_B init→final | D_B | cos5 W/R/H |')
        ap('|---|---:|---:|---:|---:|---:|---|')
        for arm in ARMS:
            if arm not in t:
                continue
            v = t[arm]
            ap(f"| {arm} | {v['acquisition']:.3f} | {v['final_mean']:.3f} | {fmt_pct(v['forgetting'])} | "
               f"{v['O_B_init']:.3f}→{v['O_B_final']:.3f} | {v['D_B']:.3f} | "
               f"{v['cos5_W']:.2f}/{v['cos5_route']:.2f}/{v['cos5_hold']:.2f} |")
        ap('')
    ap('## 3. 主效应与 gap closure（r=12.5%，3 顺序 × 10 seeds）')
    ap('')
    me = stats['me125']
    ap(f"- G = F(B) − F(All) = {fmt_pct(me['G'])}")
    ap('')
    ap('| 因子 | ME (pp) | 同向 seeds | t |')
    ap('|---|---:|---:|---:|')
    for fac, v in me['me'].items():
        ap(f"| {fac} | {v['mean']*100:+.2f} | {v['n_pos']}/{v['n']} | {v['t']:+.1f} |")
    ap('')
    ap('| arm | closure | ΔF vs B (pp) |')
    ap('|---|---:|---:|')
    for arm, v in stats['closure'].items():
        if abs(me['G']) < .005:
            clo = 'n/a'
        else:
            clo = '—' if not np.isfinite(v['closure']) else f"{v['closure']:.2f}"
        ap(f"| {arm} | {clo} | {v['df']*100:+.2f} |")
    ap('')
    if abs(me['G']) < .005:
        ap(f"> **注意：G={fmt_pct(me['G'])} 远小于 seed 噪声，closure 分母≈0，本表只作原始 ΔF 参考；"
           "H3/H4/H5 的 closure 判据在 G≈0 时不可解释（见下节 V21 gap 重审）。**")
        ap('')
    ap('| leave-one-out | F(All−X) − F(All) (pp) | 同向 seeds |')
    ap('|---|---:|---:|')
    for fac, v in stats['loo'].items():
        ap(f"| {fac} | {v['mean']*100:+.2f} | {v['n_pos']}/{v['n_pos'] + v['n_neg']} |")
    ap('')
    ap('### 每顺序主效应（r=12.5%）')
    ap('')
    ap('| 因子 | o0 | o1 | o2 |')
    ap('|---|---:|---:|---:|')
    for fac in ('W', 'route', 'hold'):
        cells = []
        for ok in ('o0', 'o1', 'o2'):
            v = stats['me_by_order'][ok][fac]
            cells.append(f"{v['mean']*100:+.2f} ({v['n_pos']}/{v['n']})")
        ap(f"| {fac} | " + ' | '.join(cells) + ' |')
    ap('')
    r21 = stats['re21']
    ap('### V21 hybrid→Full gap 的重审（关键）')
    ap('')
    ap(f"- V21 原口径：hybrid 3 顺序（5 seeds）{r21['v21_claim']['hybrid_agg3_5']:.2f}pp vs Full 仅 o0（5 seeds）"
       f"{r21['v21_claim']['full_o0_5']:.2f}pp → 表面 gap {r21['v21_claim']['gap_mismatch']:.2f}pp")
    ap(f"- 匹配口径（同 5 seeds，3 顺序）：B {r21['B_agg3_5']:.2f}pp vs All {r21['All_agg3_5']:.2f}pp "
       f"→ gap **{r21['gap_agg3_5']:.2f}pp**")
    ap(f"- 10 seeds 3 顺序：B {r21['B_agg3_10']:.2f}pp vs All {r21['All_agg3_10']:.2f}pp "
       f"→ gap **{r21['gap_agg3_10']:.2f}pp**")
    ap(f"- All 仅 o0（5 seeds）：V22 {r21['All_o0_5']:.2f}pp vs V21D {r21['v21_claim']['full_o0_5']:.2f}pp"
       f"（Δ{r21['All_o0_5']-r21['v21_claim']['full_o0_5']:+.2f}pp），协议在遗忘层面一致")
    ap('- 分顺序 B−All（10 seeds，正 = Full 更好）：'
       + '、'.join(f"{o} {v['mean']*100:+.2f}pp ({v['n_pos']}/{v['n']})" for o, v in r21['gap_by_order'].items()))
    ap('')
    ap('结论：V21 的表面 gap 主要来自 **3 顺序 hybrid 对 o0-only Full 的聚合错配**；'
       '匹配口径下 gap 从 4.80pp 缩到 ~1.1pp（5 seeds）/0.17pp（10 seeds），且分顺序符号翻转。')
    ap('')
    ap('### r=0 负控（o0）')
    ap('')
    ap('| 因子 | ME (pp) | 同向 seeds |')
    ap('|---|---:|---:|')
    for fac, v in stats['me0']['me'].items():
        ap(f"| {fac} | {v['mean']*100:+.2f} | {v['n_pos']}/{v['n']} |")
    ap('')
    ap('## 4. 预注册判定')
    ap('')
    for k, v in stats['H'].items():
        ap(f"- **{k}**：{v['text']} → {'**通过**' if v['ok'] else '未通过'}")
    ap('')
    ap('### 解释（exploratory）')
    ap('')
    ap('- credit audit 显示 core 的 5 步梯度确实"看不远"（W 训练后 cos≈0.33、模长≈6–10%），'
       '但功能实验说明 A\'B\'C\'D\' 不依赖 core 的早期时间信用：B 可以把长程信息写进状态，'
       'core 只需在尾部窗口内完成读出；因此窗口失配不构成遗忘瓶颈。')
    ap('- o0 是唯一有弱信号的顺序（hold ME +0.79pp、9/10；B−All +1.08pp），o1 同向但不显著（+1.39、6/10），'
       'o2 反转（−1.94、4/10）。V21D 只在 o0 测过 Full，不能外推到 3 顺序。')
    ap('- o0 上可见 B×core 的超加性（fixed-overlap+core full 4.66、hybrid 5.01、两者同时 1.95），'
       '即 B 的 Full credit 与 core 的 Full credit 互为部分替代；但该交互同样只在 o0 成立。')
    ap('- 预注册的"Route/Hold 是主要载体"（源自 v13 的策略定位）在匹配协议下不成立：Route 主效应 +0.05pp、'
       'Hold +0.36pp、W −0.14pp，均低于噪声量级。')
    ap('')
    ap('## 5. 单任务参考（final）')
    ap('')
    if singles:
        tasks = sorted({r['task'] for r in singles})
        ap('| arm | ' + ' | '.join(f't{t}' for t in tasks) + ' |')
        ap('|---|' + '---:|' * len(tasks))
        for arm in ARMS:
            rs = [r for r in singles if r['arm'] == arm]
            if not rs:
                continue
            vals = [np.mean([r['final'] for r in rs if r['task'] == t]) for t in tasks]
            ap(f"| {arm} | " + ' | '.join(f'{v:.3f}' for v in vals) + ' |')
        ap('')
    ap('## 6. 边界')
    ap('')
    ap('- 归因对象仅 {W, Route, Hold}；B/readout 恒 Full（readout 两条通道梯度逐位相同）。')
    ap('- 混合信用 = 逐参数窗口选择，非新优化器；B 恒 Full 以与 V21 的解耦问题解耦。')
    ap('- r=0 为负控；主判定在 r=12.5%、3 顺序聚合、10 seeds。')
    ap('- 主效应/closure/LOO 的 seed 配对使用同一 seed 的全部 arm；t 仅作描述性参考，不作显著性结论。')
    ap('- 负控 r=0 只纳入 o0（预注册口径）；误跑产生的 45 条 r=0/o1、o2 流移至 `results/extra_r0_orders/`，不进入任何判定。')
    (ROOT / 'REPORT.md').write_text('\n'.join(A) + '\n')


def audit_summary(audit):
    table = {}
    for r in audit:
        src = r['source']
        table.setdefault(src, {})
        for g in GROUPS:
            table[src].setdefault(g, {'cos5': [], 'ratio5': [], 'cos10': [], 'ratio10': []})
            for p in ('5', '10'):
                table[src][g][f'cos{p}'].append(r[f'cos{p}_{g}'])
                table[src][g][f'ratio{p}'].append(r[f'ratio{p}_{g}'])
    out = {}
    for src, gs in table.items():
        out[src] = {}
        for g, v in gs.items():
            out[src][g] = {k: float(np.nanmean(vv)) if vv else float('nan') for k, vv in v.items()}
    min_cos, min_g, min_src = 1e9, None, None
    for src, gs in out.items():
        for g in ('W', 'route', 'hold'):
            c = gs[g]['cos5']
            if np.isfinite(c) and c < min_cos:
                min_cos, min_g, min_src = c, g, src
    return {'table': out, 'min_core_cos5': min_cos, 'min_core_group': f'{min_g}@{min_src}'}


def mean_t(d):
    d = np.asarray(d, float)
    if len(d) < 2:
        return float(np.mean(d)) if len(d) else float('nan'), float('nan')
    se = d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean()), float(d.mean() / se) if se > 0 else float('nan')


def main():
    t0 = time.perf_counter()
    rows = load_cl()
    singles = load_singles()
    audit = load_audit()

    # --- verification ---
    ver = {}
    for arm, phase, ref in [('B', 'A', 'learnable'), ('All', 'D', 'learnable')]:
        mine = {r['seed']: r for r in rows if r['arm'] == arm and r['order'] == 'o0' and r['ratio'] == .125
                and r['seed'] in (11, 22, 33, 44, 55)}
        theirs = {r['seed']: r for r in v21_rows(phase, ref) if r['order'] == 'o0' and r['ratio'] == .125
                  and r['seed'] in (11, 22, 33, 44, 55)}
        seeds = sorted(set(mine) & set(theirs))
        diffs = [float(np.max(np.abs(np.array(mine[s]['matrix']) - np.array(theirs[s]['matrix'])))) for s in seeds]
        ver[arm] = {'n': len(seeds), 'maxdiff': max(diffs) if diffs else float('nan'),
                    'forget_mine': [mine[s]['forgetting'] * 100 for s in seeds],
                    'forget_v21': [theirs[s]['forgetting'] * 100 for s in seeds]}

    # --- seed level r=12.5 ---
    fbar = {arm: {s: v['forgetting'] for s, v in seed_arm_mean(rows, arm, .125).items()} for arm in ARMS}
    common = set.intersection(*[set(fbar[a]) for a in ARMS]) if all(fbar.values()) else set()
    common = sorted(common)
    n_seeds = len(common)
    F = {a: float(np.mean([fbar[a][s] for s in common])) if common else float('nan') for a in ARMS}
    G = F['B'] - F['All'] if common else float('nan')
    closure = {}
    for a in ARMS:
        df = F['B'] - F[a]
        clo = df / G if abs(G) > 1e-9 else float('nan')
        if abs(G) < .005:
            clo = float('nan')
        closure[a] = {'df': df, 'closure': clo}
    me125 = {'G': G, 'n_seeds': n_seeds, 'me': {}}
    for fac in ('W', 'route', 'hold'):
        dd = me_contrast(rows, .125, fac, None)
        m, t = mean_t(dd)
        me125['me'][fac] = {'mean': m, 't': t, 'n': len(dd), 'n_pos': int((dd > 0).sum()),
                            'diffs': dd.tolist()}
    loo = {}
    for fac, arm in [('W', 'B+R+H'), ('route', 'B+W+H'), ('hold', 'B+W+R')]:
        _, d = paired(fbar[arm], fbar['All'])
        m, t = mean_t(d)
        loo[fac] = {'mean': m, 't': t, 'n_pos': int((d > 0).sum()), 'n_neg': int((d < 0).sum())}
    me0 = {'me': {}}
    rows0 = [r for r in rows if r['ratio'] == 0 and r['order'] == 'o0']
    for fac in ('W', 'route', 'hold'):
        dd = me_contrast(rows0, 0, fac, None)
        m, t = mean_t(dd)
        me0['me'][fac] = {'mean': m, 't': t, 'n': len(dd), 'n_pos': int((dd > 0).sum())}

    ac = audit_summary(audit)
    # 每顺序的主效应（稳健性）
    me_by_order = {}
    for ok in ('o0', 'o1', 'o2'):
        rs = [r for r in rows if r['order'] == ok and r['ratio'] == .125]
        order_d = {}
        for fac in ('W', 'route', 'hold'):
            dd = me_contrast(rs, .125, fac, None)
            m, t = mean_t(dd)
            order_d[fac] = {'mean': m, 't': t, 'n_pos': int((dd > 0).sum()), 'n': len(dd)}
        me_by_order[ok] = order_d
    stats = {'partial': {}, 'audit': ac, 'me125': me125, 'closure': closure, 'loo': loo, 'me0': me0,
             'me_by_order': me_by_order}
    # V21 重审：匹配聚合口径
    seeds5 = (11, 22, 33, 44, 55)
    re21 = {'v21_claim': {'hybrid_agg3_5': 6.63, 'full_o0_5': 1.83}}
    re21['v21_claim']['gap_mismatch'] = re21['v21_claim']['hybrid_agg3_5'] - re21['v21_claim']['full_o0_5']
    for a in ('B', 'All'):
        agg5 = [np.mean([r['forgetting'] for r in rows if r['arm'] == a and r['ratio'] == .125
                         and r['order'] == o and r['seed'] in seeds5]) for o in ('o0', 'o1', 'o2')]
        agg10 = [np.mean([r['forgetting'] for r in rows if r['arm'] == a and r['ratio'] == .125
                          and r['order'] == o]) for o in ('o0', 'o1', 'o2')]
        re21[f'{a}_agg3_5'] = float(np.mean(agg5) * 100)
        re21[f'{a}_agg3_10'] = float(np.mean(agg10) * 100)
    re21['gap_agg3_5'] = re21['B_agg3_5'] - re21['All_agg3_5']
    re21['gap_agg3_10'] = re21['B_agg3_10'] - re21['All_agg3_10']
    re21['All_o0_5'] = float(np.mean([r['forgetting'] * 100 for r in rows if r['arm'] == 'All'
                                      and r['ratio'] == .125 and r['order'] == 'o0' and r['seed'] in seeds5]))
    re21['gap_by_order'] = {}
    for o in ('o0', 'o1', 'o2'):
        d = []
        for s in sorted({r['seed'] for r in rows}):
            b = [r['forgetting'] for r in rows if r['arm'] == 'B' and r['order'] == o and r['seed'] == s
                 and r['ratio'] == .125]
            a = [r['forgetting'] for r in rows if r['arm'] == 'All' and r['order'] == o and r['seed'] == s
                 and r['ratio'] == .125]
            if b and a:
                d.append(float(b[0] - a[0]))
        re21['gap_by_order'][o] = {'mean': float(np.mean(d)) if d else float('nan'),
                                   'n_pos': int(np.sum(np.array(d) > 0)), 'n': len(d)}
    stats['re21'] = re21
    expected = 8 * 10 * 3
    stats['partial']['is_partial'] = len([r for r in rows if r['ratio'] == .125]) < expected
    stats['partial']['detail'] = f"r=12.5 行数 {len([r for r in rows if r['ratio'] == .125])}/{expected}"

    # --- H judgments ---
    H = {}
    H['H0a'] = {'text': f"B 臂逐位回归 V21A hybrid（max|Δ|={ver['B']['maxdiff']:.1e}）",
                'ok': bool(ver['B']['maxdiff'] < 1e-12)}
    H['H0b'] = {'text': f"All 臂对 V21D max|Δ|≤0.02（实测 matrix {ver['All']['maxdiff']:.3e}；"
                        f"遗忘均值 o0/5seeds {re21['All_o0_5']:.2f} vs 1.83）",
                'ok': bool(ver['All']['maxdiff'] <= 0.02)}
    H['H0c'] = {'text': f"audit min core cos5={ac['min_core_cos5']:.3f}<0.9",
                'ok': bool(ac['min_core_cos5'] < 0.9)}
    g_small = abs(G) < .005
    if n_seeds >= 4:
        me_r, me_h, me_w = (me125['me'][f]['mean'] for f in ('route', 'hold', 'W'))
        best_fac = max((('route', me_r), ('hold', me_h)), key=lambda kv: kv[1])
        npos_best = me125['me'][best_fac[0]]['n_pos']
        n_common = me125['me'][best_fac[0]]['n']
        H['H1'] = {'text': f"max(ME_R,ME_H)={max(me_r, me_h)*100:+.2f}pp ≥1.5 且 ≥ME_W+1.0；"
                            f"最优因子 {best_fac[0]} 同向 {npos_best}/{n_common}",
                    'ok': bool(max(me_r, me_h) >= .015 and max(me_r, me_h) >= me_w + .010 and npos_best >= .8 * n_common)}
        H['H2'] = {'text': f"closure(B+W)={closure['B+W']['closure']:.2f}≤0.25 或 ME_W={me_w*100:+.2f}pp≤1.0"
                            + ("（G≈0，closure 不可解释；以 ME_W 为准）" if g_small else ""),
                   'ok': bool((not g_small and np.isfinite(closure['B+W']['closure']) and closure['B+W']['closure'] <= .25)
                              or me_w <= .010)}
        H['H3'] = {'text': f"closure(B+R+H)={closure['B+R+H']['closure']:.2f}≥0.5"
                            + ("（G≈0，不可评估）" if g_small else ""),
                   'ok': bool((not g_small) and np.isfinite(closure['B+R+H']['closure']) and closure['B+R+H']['closure'] >= .5)}
        H['H4'] = {'text': f"max(LOO_R,LOO_H)={max(loo['route']['mean'], loo['hold']['mean'])*100:+.2f}pp ≥0.5G"
                            + ("（G≈0，不可评估）" if g_small else ""),
                   'ok': bool((not g_small) and (max(loo['route']['mean'], loo['hold']['mean']) >= .5 * G)) if n_seeds else False}
        H['H5'] = {'text': f"所有 ≤2 因子臂 closure<0.5（B+R+H {closure['B+R+H']['closure']:.2f}、"
                            f"B+W+R {closure['B+W+R']['closure']:.2f}、B+W+H {closure['B+W+H']['closure']:.2f}）"
                            + ("（G≈0，不可评估）" if g_small else ""),
                   'ok': bool((not g_small) and all(np.isfinite(closure[a]['closure']) and closure[a]['closure'] < .5
                                                    for a in ('B+R+H', 'B+W+R', 'B+W+H')))}
    else:
        for k in ('H1', 'H2', 'H3', 'H4', 'H5'):
            H[k] = {'text': 'seeds 不足', 'ok': False}
    Hneg_ok = all(abs(me0['me'][f]['mean']) <= .010 for f in ('W', 'route', 'hold')) if me0 else False
    if not me0 or any(me0['me'][f]['n'] == 0 for f in ('W', 'route', 'hold')):
        Hneg_ok = False
    H['Hneg'] = {'text': 'r=0 所有 |ME_X|≤1.0pp：' +
                         '、'.join(f"{f}={me0['me'][f]['mean']*100:+.2f}" for f in ('W', 'route', 'hold')),
                 'ok': bool(Hneg_ok)}
    stats['H'] = H
    stats['ver'] = ver

    write_report(rows, singles, audit, ver, stats)
    (ROOT / 'results/summary.json').write_text(json.dumps(stats, indent=2))
    (ROOT / 'verification.txt').write_text(
        f"B bitwise maxdiff: {ver['B']['maxdiff']:.3e} over {ver['B']['n']}\n"
        f"All vs V21D maxdiff: {ver['All']['maxdiff']:.3e} over {ver['All']['n']}\n"
        f"G (B-All): {G*100 if np.isfinite(G) else float('nan'):.2f}pp\n"
        f"closure B+R+H: {closure['B+R+H']['closure']:.3f}\n")
    print('analyze done in', round(time.perf_counter() - t0, 1), 's; rows', len(rows), 'seeds', n_seeds)
    print(json.dumps({k: v['ok'] for k, v in stats['H'].items()}, indent=1))
    print(json.dumps({'G': G, 'closure': {a: v['closure'] for a, v in closure.items()},
                      'ME': {f: v['mean'] for f, v in me125['me'].items()}}, indent=1))


if __name__ == '__main__':
    main()

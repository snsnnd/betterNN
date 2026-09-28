"""V25 Phase 0 汇总：A/B/F 矩阵、位移/梯度/landscape、Q1–Q3 判定、REPORT.md。

只读 v25/results/audit 与 v24 锚点 JSON，不改动任何结果。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
V24 = ROOT.parent / 'flow_mvp_v24'
T, TASK = 80, 0
KERNELS = ['single', 'burst3', 'burst5', 'decay-fast', 'decay-slow']
ARMS = ['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'F1', 'F2', 'F3']


def load_audit(aud):
    return [json.loads(f.read_text()) for f in sorted(aud.glob('audit_*.json'))]


def load_v24(sub, pat):
    d = V24 / 'results' / sub
    return [json.loads(f.read_text()) for f in sorted(d.glob(pat))] if d.exists() else []


def mean_se(xs):
    xs = [x for x in xs if np.isfinite(x)]
    if not xs:
        return float('nan'), float('nan')
    m = float(np.mean(xs))
    se = float(np.std(xs, ddof=1) / np.sqrt(len(xs))) if len(xs) > 1 else 0.0
    return m, se


def arm_rows(rows, arm, ep=None):
    rs = [r for r in rows if r['arm'] == arm and r['T'] == T and r['task'] == TASK]
    if ep is not None:
        rs = [r for r in rs if r['epochs'] == ep]
    return rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/audit')
    args = ap.parse_args()
    aud = ROOT / args.out
    rows = load_audit(aud)
    if not rows:
        print('no audit results in', aud)
        return
    v24_bank = [r for r in load_v24('bank_delay', f'bank_delay_T{T}_t{TASK}_*.json')]
    v24_fixed = [r for r in load_v24('delay', f'delay_*_T{T}_t{TASK}_*.json')]
    hdr = f'T={T} task={TASK}'
    A = []
    ap = A.append
    ap('# 第二十五轮（V24D-Phase 0）：Temporal Strategy 可发现性审计')
    ap('')
    ap(f'协议：{hdr}、B fixed-disjoint、core K=5、scheduler hybrid full；'
       'A1 与 V24 `bank_delay` 同代码路径（应在同 seed 上逐位一致），65ep fixed 锚点读 V24。')
    ap('')

    # ---------- 回归 ----------
    ap('## 1. 回归锚点')
    ap('')
    reg = {}
    for r in arm_rows(rows, 'A1'):
        if r['epochs'] != 65:
            continue
        ref = [v for v in v24_bank if v['seed'] == r['seed']]
        if ref:
            reg[r['seed']] = abs(r['final'] - ref[0]['final'])
    if reg:
        ap(f"- A1 vs V24 bank_delay 同 seed：n={len(reg)}，max|Δfinal|="
           f"{max(reg.values()):.3e}（期望 0），per-seed={ {k: round(v, 6) for k, v in reg.items()} }")
    else:
        ap('- 无 A1@65ep 结果或 V24 锚点，跳过回归（smoke/部分运行属正常）。')
    ap('')

    # ---------- 矩阵 ----------
    a1 = {r['seed']: r for r in arm_rows(rows, 'A1')}
    ap('## 2. 初始化/预算矩阵')
    ap('')
    ap('| arm | 参数化/初始化 | ep | n | final | best-val | Δ vs A1（同 seed） | 位移 |')
    ap('|---|---|---:|---:|---|---:|---|---|')
    summ = {}
    for arm in ARMS:
        rs = arm_rows(rows, arm)
        if not rs:
            continue
        fin_m, fin_se = mean_se([r['final'] for r in rs])
        bv_m, bv_se = mean_se([r['best_val'] for r in rs])
        deltas = [r['final'] - a1[r['seed']]['final'] for r in rs if r['seed'] in a1]
        d_m, _ = mean_se(deltas)
        if arm.startswith('A'):
            desc = f"bank/{rs[0]['init']}"
            disp = mean_se([r['displacement']['d_alpha_L1_mean'] for r in rs])[0]
            disp_s = f"Δα_L1={disp:.2f}"
        elif arm.startswith('B'):
            desc = f"shared λ0={rs[0]['lam0']}"
            disp = mean_se([r['displacement']['d_lambda'] for r in rs])[0]
            disp_s = f"|Δλ|={disp:.2f}"
        else:
            desc = f"fixed/{rs[0]['kernel']}"
            disp_s = '—'
        ep = rs[0]['epochs']
        suffix = '±%.3f' % fin_se
        ap(f"| {arm} | {desc} | {ep} | {len(rs)} | {fin_m:.3f}{suffix} | {bv_m:.3f} | "
           f"{d_m*100:+.1f}pp | {disp_s} |")
        summ[arm] = {'final': fin_m, 'best': bv_m, 'delta_vs_a1': d_m, 'n': len(rs),
                     'end': rs[0]['profile_end'] if 'profile_end' in rs[0] else None,
                     'disp': disp if not arm.startswith('F') else None}

    # ---------- shared λ 梯度 ----------
    ap('')
    ap('## 3. shared λ：初始梯度与位移')
    ap('')
    ap('| arm | λ0 | λ_end(mean) | |Δλ| | final | ∂L/∂λ (init, mean±std) | batch<0 |')
    ap('|---|---:|---:|---:|---|---|---:|')
    gsum = {}
    for arm in ('B1', 'B2', 'B3'):
        rs = arm_rows(rows, arm)
        if not rs:
            continue
        le = mean_se([r['profile_end']['A']['lambda'] for r in rs])[0]
        dl = mean_se([r['displacement']['d_lambda'] for r in rs])[0]
        lm = mean_se([r['grad_init']['full']['lambda_mean'] for r in rs])[0]
        ls = mean_se([r['grad_init']['full']['lambda_std'] for r in rs])[0]
        nneg = sum(r['grad_init']['full']['n_negative'] for r in rs)
        nb = sum(len(r['grad_init']['full']['per_batch_lambda']) for r in rs)
        ap(f"| {arm} | {rs[0]['lam0']} | {le:.3f} | {dl:.3f} | "
           f"{summ[arm]['final']:.3f} | {lm:+.4f}±{ls:.4f} | {nneg}/{nb} |")
        gsum[arm] = {'lambda_mean': lm, 'lambda_std': ls, 'n_neg': nneg, 'n_batch': nb}

    # ---------- landscape ----------
    def start_x(arm, mode, r):
        """该 arm 训练起点在扫描轴上的位置：bank 按初始化核所在端点；shared 按 λ0。"""
        if mode == 'lambda':
            return r['lam0']
        if arm in ('A1', 'A4'):
            return 0.0
        if arm == 'A2' and mode == 'burst5':
            return 1.0
        if arm == 'A3' and mode == 'decay-slow':
            return 1.0
        return None

    ap('')
    ap('## 4. 条件 landscape（冻结参数，init/end 两查）')
    ap('')
    ap('| arm | 扫描 | x*_init | minL_init | gap_init | x*_end | minL_end | gap_end |')
    ap('|---|---|---:|---:|---:|---:|---:|---:|')
    for arm in ARMS:
        rs = arm_rows(rows, arm)
        if not rs or arm.startswith('F'):
            continue
        for mode in rs[0]['scan_init'].keys():
            m_rows = [r for r in rs if mode in r.get('scan_end', {})]
            if not m_rows:
                continue
            xi = mean_se([min(r['scan_init'][mode], key=lambda p: p['loss'])['x'] for r in m_rows])[0]
            li = mean_se([min(p['loss'] for p in r['scan_init'][mode]) for r in m_rows])[0]
            xe = mean_se([min(r['scan_end'][mode], key=lambda p: p['loss'])['x'] for r in m_rows])[0]
            le = mean_se([min(p['loss'] for p in r['scan_end'][mode]) for r in m_rows])[0]

            def gap(scan_rows, key):
                vals = []
                for r in scan_rows:
                    sx = start_x(arm, mode, r)
                    if sx is None:
                        continue
                    pts = r[key][mode]
                    l0 = min(pts, key=lambda p: abs(p['x'] - sx))['loss']
                    vals.append(l0 - min(p['loss'] for p in pts))
                return mean_se(vals)[0]

            gi, ge = gap(m_rows, 'scan_init'), gap(m_rows, 'scan_end')

            def fmt(v, spec):
                return '—' if not np.isfinite(v) else format(v, spec)
            ap(f"| {arm} | {mode} | {xi:.2f} | {li:.4f} | {fmt(gi, '+.4f')} | {xe:.2f} | "
               f"{le:.4f} | {fmt(ge, '+.4f')} |")

    # ---------- 判定 ----------
    ap('')
    ap('## 5. Phase 0 判定')
    ap('')
    verdict = {}

    def best_fixed65(seed):
        vals = [r['final'] for r in v24_fixed if r['seed'] == seed]
        return max(vals) if vals else float('nan')

    # Q1
    q1 = {}
    for arm in ('A2', 'A3'):
        rs = arm_rows(rows, arm)
        if not rs:
            continue
        stay = collapse = 0
        for r in rs:
            fb = best_fixed65(r['seed'])
            a1v = a1.get(r['seed'], {}).get('final', float('nan'))
            d = r['displacement']['d_alpha_L1_mean']
            if d < 0.2 and r['final'] >= fb - 0.02 and r['final'] >= a1v + 0.02:
                stay += 1
            al_end = [r['profile_end'][e]['alpha'] for e in ('A', 'B')]
            l1_single = float(np.mean([np.abs(np.array(a) - np.array([1, 0, 0, 0, 0])).sum()
                                       for a in al_end]))
            if l1_single < 0.2 or r['final'] <= fb - 0.05:
                collapse += 1
        q1[arm] = {'stay': stay, 'collapse': collapse, 'n': len(rs)}
    verdict['Q1'] = q1
    ap(f"- Q1（init basin）A2/A3 停留/塌回计数：{q1}（stay 需同时满足位移<0.2、≥best fixed−2pp、≥A1+2pp）")

    # Q2
    q2 = {}
    for arm in ('B1', 'B2', 'B3'):
        rs = arm_rows(rows, arm)
        if not rs:
            continue
        moves = stuck = 0
        for r in rs:
            dl = r['displacement']['d_lambda']
            a1v = a1.get(r['seed'], {}).get('final', float('nan'))
            if dl >= 0.2 and r['final'] >= a1v + 0.02:
                moves += 1
            if dl <= 0.05 and abs(r['final'] - a1v) <= 0.01:
                stuck += 1
        q2[arm] = {'moves': moves, 'stuck': stuck, 'n': len(rs)}
    verdict['Q2'] = q2
    ap(f"- Q2（shared λ）会动/不动计数：{q2}")

    # Q3
    q3 = {}
    for arm, base, ctrl in (('A4', 'A1', 'F1'), ('B3', 'B1', 'F1')):
        rs = arm_rows(rows, arm)
        if not rs:
            continue
        base_map = {r['seed']: r for r in arm_rows(rows, base)}
        ctrl_map = {r['seed']: r for r in arm_rows(rows, ctrl)}
        gain_a1 = [r['final'] - base_map[r['seed']]['final'] for r in rs if r['seed'] in base_map]
        gain_f = [r['final'] - ctrl_map[r['seed']]['final'] for r in rs if r['seed'] in ctrl_map]
        budget = sum(1 for a, b in zip(gain_a1, gain_f) if a >= 0.02 and b >= 0.02)
        non = sum(1 for a in gain_a1 if a <= 0.01)
        q3[arm] = {'gain_vs_base': gain_a1, 'gain_vs_fixed': gain_f,
                   'budget_seeds': budget, 'non_budget_seeds': non, 'n': len(rs)}
    verdict['Q3'] = q3
    ap(f"- Q3（预算）A4/B3：{q3}")

    # 决策映射
    lines = []
    if q1:
        stay_total = sum(v['stay'] for v in q1.values())
        collapse_total = sum(v['collapse'] for v in q1.values())
        if stay_total >= 2:
            lines.append('A2/A3 在好核初始化下能保持 → 表示能力存在，**探索/退火（Phase 2）**优先')
        if collapse_total >= 2:
            lines.append('A2/A3 从好核塌回 → 联合训练会破坏好解，先查 hybrid 两遍协议与耦合，暂停 Phase 1/2')
    if q2:
        moved = sum(v['moves'] for v in q2.values())
        stuck = sum(v['stuck'] for v in q2.values())
        if moved >= 2:
            lines.append('shared λ 会动 → 条件化 MLP 是主要问题，**Phase 1（连续 per-event λ）**优先')
        if stuck >= 2:
            lines.append('shared λ 也不动 → 局部梯度弱/平，**探索/退火**优先')
    if q3:
        for arm, v in q3.items():
            if v['budget_seeds'] >= 2:
                lines.append(f'{arm}: 500ep 有增益 → 先排除预算/协同适应慢，再解释参数化')
            elif v['non_budget_seeds'] >= 2:
                lines.append(f'{arm}: 500ep 无增益 → 预算不是主因')
    verdict['routes'] = lines or ['（无足够结果）']
    ap('- 决策映射（可组合）：')
    for l in verdict['routes']:
        ap(f"  - {l}")

    # ---------- exploratory：预算混淆 ----------
    ap('')
    ap('## 5.5 Exploratory：预算混淆检查（非预注册，事后诊断）')
    ap('')
    ap('| seed | single65 | best65（kernel） | gap65 | F1 single500 | F2 burst5 | F3 dslow500 | best500−F1 |')
    ap('|---|---:|---|---:|---:|---:|---:|---:|')
    for s in sorted({r['seed'] for r in rows if r['arm'] == 'A1'}):
        kk = {k: [r['final'] for r in v24_fixed if r['seed'] == s and r['kernel'] == k]
              for k in KERNELS}
        kk = {k: v[0] for k, v in kk.items() if v}
        if not kk:
            continue
        bk = max(kk, key=kk.get)
        f = {a: next((r['final'] for r in arm_rows(rows, a) if r['seed'] == s), float('nan'))
             for a in ('F1', 'F2', 'F3')}
        b500 = max(f['F1'], f['F2'], f['F3'])
        ap(f"| {s} | {kk['single']:.3f} | {kk[bk]:.3f}（{bk}） | "
           f"{(kk[bk] - kk['single']) * 100:+.1f}pp | {f['F1']:.3f} | {f['F2']:.3f} | "
           f"{f['F3']:.3f} | {(b500 - f['F1']) * 100:+.1f}pp |")
    ap('')
    ap('观察（exploratory）：65ep 上 best fixed 相对 single 的领先在 500ep 收缩到 ±4pp 以内，'
       '且 A4/B3 与 F1 齐平 → delay T80/task0 上“selector 找不到 fixed kernel 收益”'
       '至少部分是**训练预算/收敛速度混淆**，而不是最终可达性能差异。'
       'A2/A3 的 profile 显示终点仍在初始核族（burst5 / decay-slow 占优，未回到 single）；'
       '“塌回”标签来自与 65ep best fixed 的性能比较，受同一混淆影响。')
    ap('')
    ap('## 6. 备注')
    ap('')
    ap('- landscape 是**条件**曲线（core 按自身策略训练后冻结），只作局部几何证据；联合可达性看 V24 Phase A 与 F* arms。')
    ap('- 3 seeds 下阈值附近不做强结论；若判定落在边界，补 44/55 两个 seed。')
    ap('- 本次未实现 Phase 1/2（见 PLAN §6）；是否进入由上述映射决定。')

    (ROOT / 'REPORT.md').write_text('\n'.join(A) + '\n')
    (aud / 'summary.json').write_text(json.dumps(verdict, indent=2, default=str))
    print('\n'.join(A[-30:]))
    print(f'\nwrote REPORT.md / {aud.name}/summary.json')


if __name__ == '__main__':
    sys.exit(main())

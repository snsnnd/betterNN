"""V26 汇总：Acc500 / Acc451-500 / E90-98 / AUC、V24@65 回归、H-cap/H-opt/H-null 判定。"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
V24 = ROOT.parent / 'flow_mvp_v24'
KERNELS = ['single', 'burst5', 'decay-slow']
SEEDS = [11, 22, 33, 44, 55]
BUDGET_POINTS = (25, 50, 65, 100, 200, 300, 500)


def load(dirpath):
    return [json.loads(f.read_text()) for f in sorted(dirpath.glob('rebase_*.json'))]


def metrics(row):
    c = row['curve']
    val = {e['epoch']: e['val'] for e in c}
    eps = sorted(val)
    last = max(eps)
    test_last = next((e.get('test') for e in c if e['epoch'] == last), None)

    def first(th):
        for i, e in enumerate(eps):
            if val[e] >= th and all(val[x] >= th for x in eps[i + 1:i + 3]):
                return e
        return None

    return {'acc': val[last], 'acc451': float(np.mean([val[e] for e in eps if e >= last - 49])),
            'test': test_last, 'auc': float(np.trapezoid([val[e] for e in eps], eps) / last),
            'E90': first(.90), 'E95': first(.95), 'E98': first(.98)}


def med(vals):
    vals = [v for v in vals if v is not None]
    return float(np.median(vals)) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dirs', nargs='+', default=None)
    args = ap.parse_args()
    dirs = [ROOT / d for d in args.dirs] if args.dirs else sorted(ROOT.glob('results/rebase_*'))
    rows = [r for d in dirs for r in load(d)]
    if not rows:
        print('no rebase results in', [str(d) for d in dirs])
        return
    for r in rows:
        r['m'] = metrics(r)
    Ts = sorted({r['T'] for r in rows})
    A = []
    ap_ = A.append
    ap_('# 第二十六轮：Budget-Matched Temporal Write Re-baseline')
    ap_('')
    ap_('协议与 V24 Phase A 一致（B fixed-disjoint、core K=5、Adam lr=.003、batch 128），'
       '只换 write kernel；500 epochs、tasks 0–3、seeds 11/22/33/44/55。'
       f'数据目录：{", ".join(d.name for d in dirs)}。')
    ap_('')

    # -------- 回归 --------
    ap_('## 1. V24@65 回归（test@65 vs V24 delay_* final）')
    ap_('')
    reg, missing = [], 0
    for r in rows:
        e65 = next((c for c in r['curve'] if c['epoch'] == 65), None)
        if e65 is None or 'test' not in e65:
            missing += 1
            continue
        p = V24 / 'results' / 'delay' / f"delay_{r['kernel']}_T{r['T']}_t{r['task']}_{r['seed']}.json"
        if not p.exists():
            missing += 1
            continue
        v = json.loads(p.read_text())
        reg.append(abs(e65['test'] - v['final']))
    if reg:
        ap_(f"- n={len(reg)} cells，max|Δtest@65|={max(reg):.3e}（期望 0）"
           + (f"，{missing} cells 缺 V24 锚点" if missing else ''))
    else:
        ap_('- 无可用回归 cell（部分运行属正常）。')
    ap_('')

    # -------- 汇总 --------
    ap_('## 2. 各 kernel 汇总（均值 ± se，cells = tasks × seeds）')
    ap_('')
    ap_('| T | kernel | Acc500(val) | Acc451-500 | test@500 | AUC | E90 | E95 | E98 | 达到95 |')
    ap_('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    summ = {}
    for T in Ts:
        for k in KERNELS:
            rs = [r for r in rows if r['T'] == T and r['kernel'] == k]
            if not rs:
                continue
            def ms(key):
                xs = [r['m'][key] for r in rs if r['m'][key] is not None]
                return np.mean(xs), (np.std(xs, ddof=1) / np.sqrt(len(xs)) if len(xs) > 1 else 0.)
            acc_m, acc_se = ms('acc')
            a451_m, _ = ms('acc451')
            t_m, _ = ms('test')
            auc_m, _ = ms('auc')
            e90 = med([r['m']['E90'] for r in rs])
            e95 = med([r['m']['E95'] for r in rs])
            e98 = med([r['m']['E98'] for r in rs])
            n95 = sum(1 for r in rs if r['m']['E95'] is not None)
            ap_(f"| {T} | {k} | {acc_m:.3f}±{acc_se:.3f} | {a451_m:.3f} | {t_m:.3f} | "
                f"{auc_m:.3f} | {e90 if e90 is not None else '—'} | "
                f"{e95 if e95 is not None else '—'} | {e98 if e98 is not None else '—'} | "
                f"{n95}/{len(rs)} |")
            summ[(T, k)] = {'acc': acc_m, 'acc451': a451_m, 'test': t_m, 'auc': auc_m,
                            'E95_med': e95, 'n95': n95, 'n': len(rs)}

    # -------- 分任务 --------
    ap_('')
    ap_('## 2.5 分任务 Acc500（seeds 均值）')
    ap_('')
    ap_('| T | task | single | burst5 | decay-slow | burst5−single |')
    ap_('|---|---:|---:|---:|---:|---:|')
    for T in Ts:
        for task in sorted({r['task'] for r in rows if r['T'] == T}):
            vals = {}
            for k in KERNELS:
                xs = [r['m']['acc'] for r in rows if r['T'] == T and r['kernel'] == k and r['task'] == task]
                vals[k] = float(np.mean(xs)) if xs else float('nan')
            ap_(f"| {T} | {task} | {vals['single']:.3f} | {vals['burst5']:.3f} | "
                f"{vals['decay-slow']:.3f} | {(vals['burst5']-vals['single'])*100:+.1f}pp |")

    # -------- pairwise --------
    ap_('')
    ap_('## 3. Pairwise（burst5−single、decay-slow−single；每 seed 先对 4 task 取平均）')
    ap_('')
    ap_('| T | 对比 | ΔAcc500 | 同向 | ΔAcc451 | ΔAUC | E95_single | E95_other |')
    ap_('|---|---|---:|---:|---:|---:|---:|---:|')
    verdict = {}
    for T in Ts:
        for k1 in ('burst5', 'decay-slow'):
            k0 = 'single'
            rs1 = {(r['task'], r['seed']): r for r in rows if r['T'] == T and r['kernel'] == k1}
            rs0 = {(r['task'], r['seed']): r for r in rows if r['T'] == T and r['kernel'] == k0}
            cells = sorted(set(rs1) & set(rs0))
            if not cells:
                continue
            per_seed = {}
            for task, seed in cells:
                per_seed.setdefault(seed, []).append((task, rs1[(task, seed)], rs0[(task, seed)]))
            d_acc = [np.mean([r1['m']['acc'] - r0['m']['acc'] for _, r1, r0 in v])
                     for v in per_seed.values()]
            d_a451 = [np.mean([r1['m']['acc451'] - r0['m']['acc451'] for _, r1, r0 in v])
                      for v in per_seed.values()]
            d_auc = [np.mean([r1['m']['auc'] - r0['m']['auc'] for _, r1, r0 in v])
                     for v in per_seed.values()]
            e95_1 = med([r1['m']['E95'] for v in per_seed.values() for _, r1, _ in v])
            e95_0 = med([r0['m']['E95'] for v in per_seed.values() for _, _, r0 in v])
            npos = sum(1 for d in d_acc if d > 0)
            m_acc = float(np.mean(d_acc))
            m_auc = float(np.mean(d_auc))
            ap_(f"| {T} | {k1}−{k0} | {m_acc*100:+.1f}pp | {npos}/{len(d_acc)} | "
                f"{np.mean(d_a451)*100:+.1f}pp | {m_auc*100:+.1f}pp | "
                f"{e95_0 if e95_0 is not None else '—'} | {e95_1 if e95_1 is not None else '—'} |")
            cap = (m_acc >= 0.03 and npos >= max(4, len(d_acc) - 1)) and \
                  (float(np.mean(d_a451)) >= 0.03 and sum(1 for d in d_a451 if d > 0) >= max(4, len(d_acc) - 1))
            auc_pos = sum(1 for d in d_auc if d >= 0.01)
            e95fav = 0
            for v in per_seed.values():
                h1 = med([r1['m']['E95'] for _, r1, _ in v])
                h0 = med([r0['m']['E95'] for _, _, r0 in v])
                if h1 is not None and (h0 is None or h1 <= 0.8 * h0):
                    e95fav += 1
            opt = (abs(m_acc) < 0.02) and (auc_pos >= max(4, len(d_acc) - 1) or
                                           e95fav >= max(4, len(d_acc) - 1))
            weak = (not cap) and 0.02 <= abs(m_acc) < 0.03
            verdict[(T, k1)] = {'d_acc': m_acc, 'n_pos': npos, 'n': len(d_acc),
                                'd_auc': m_auc, 'auc_pos': auc_pos, 'e95_fav': e95fav,
                                'cap': cap, 'opt': opt, 'weak': weak}

    # -------- 预算曲线 --------
    ap_('')
    ap_('## 4. 预算曲线（val acc，均值；epochs 25/50/65/100/200/300/500）')
    ap_('')
    ap_('| T | kernel | ' + ' | '.join(str(e) for e in BUDGET_POINTS) + ' |')
    ap_('|---|---|' + '---:|' * len(BUDGET_POINTS))
    for T in Ts:
        for k in KERNELS:
            rs = [r for r in rows if r['T'] == T and r['kernel'] == k]
            if not rs:
                continue
            vals = []
            for ep in BUDGET_POINTS:
                xs = [c['val'] for r in rs for c in r['curve'] if c['epoch'] == ep]
                vals.append(f"{np.mean(xs):.3f}" if xs else '—')
            ap_(f"| {T} | {k} | " + ' | '.join(vals) + ' |')

    # -------- 判定 --------
    ap_('')
    ap_('## 5. 预注册判定')
    ap_('')
    for T in Ts:
        for k1 in ('burst5', 'decay-slow'):
            v = verdict.get((T, k1))
            if not v:
                continue
            tag = ('H-cap' if v['cap'] else 'H-opt' if v['opt'] else
                   'weak（2–3pp，不判 H-cap）' if v['weak'] else 'H-null')
            ap_(f"- T={T} {k1}−single：ΔAcc500={v['d_acc']*100:+.1f}pp（{v['n_pos']}/{v['n']} 同向）、"
                f"ΔAUC={v['d_auc']*100:+.1f}pp（{v['auc_pos']}/{v['n']} ≥+1pp）、"
                f"E95 有利 {v['e95_fav']}/{v['n']} → **{tag}**")
    ap_('')
    ap_('结论映射：T=160 若 H-cap → temporal kernel 改变最终能力，支持多时间尺度状态；'
       '仅 H-opt/H-null → delayed benchmark 上主要是收敛速度或无效，delayed 线降级、'
       '主证据收缩到 Chain-select（V24 500ep 已验证）；只有 H-cap 才实现 C/D（adaptive bank）与 continuous λ。')
    ap_('')
    ap_('## 6. 边界')
    ap_('')
    ap_('- Acc451-500 与 E90/95/98/AUC 均为 val 指标，不用测试集选模；E 未达到记 None。')
    ap_('- 500 epochs 是共同预算上限，不构成“充分收敛”证明。')
    ap_('- 2–3pp 的稳定差记为 weak，不判 H-cap。')

    (ROOT / 'REPORT.md').write_text('\n'.join(A) + '\n')
    (ROOT / 'results' / 'summary.json').write_text(json.dumps(
        {f"{k[0]}_{k[1]}": v for k, v in verdict.items()}, indent=2, default=str))
    print('\n'.join(A))
    print('\nwrote REPORT.md / results/summary.json')


if __name__ == '__main__':
    sys.exit(main())

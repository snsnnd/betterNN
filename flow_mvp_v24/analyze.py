"""第二十四轮汇总：write kernel × T 网格、retention、H1–H4、REPORT.md。"""
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
V20 = (ROOT / '../flow_mvp_v20').resolve()
KERNELS = ['single', 'burst3', 'burst5', 'decay-fast', 'decay-slow']
CHAIN_KERNELS = ['single', 'burst5', 'decay-slow']


def load_dir(d):
    rows = []
    p = ROOT / d
    if p.exists():
        for f in sorted(p.glob('*.json')):
            rows.append(json.loads(f.read_text()))
    return rows


def delay_rows():
    return [r for r in load_dir('results/delay') if r.get('phase') == 'delay']


def chain_rows():
    return [r for r in load_dir('results/chain') if r.get('phase') == 'chain']


def cell(rows, kernel, T):
    return [r for r in rows if r['kernel'] == kernel and r['T'] == T]


def mean_se(v):
    v = np.asarray(v, float)
    se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return float(v.mean()), float(se)


def paired(rows, kernel, T, key='final'):
    a = {(r['task'], r['seed']): r[key] for r in cell(rows, kernel, T)}
    b = {(r['task'], r['seed']): r[key] for r in cell(rows, 'single', T)}
    keys = sorted(set(a) & set(b))
    return np.array([a[k] - b[k] for k in keys])


def paired_ret(rows, kernel, T, key='ret_half'):
    """返回 log-ratio 均值（几何均值比），对离群更稳健。"""
    a = {(r['task'], r['seed']): r['retention'][key] for r in cell(rows, kernel, T)}
    b = {(r['task'], r['seed']): r['retention'][key] for r in cell(rows, 'single', T)}
    keys = sorted(set(a) & set(b))
    rel = []
    for k in keys:
        if a[k] > 0 and b[k] > 0 and np.isfinite(a[k]) and np.isfinite(b[k]):
            rel.append(np.log(a[k]) - np.log(b[k]))
    return np.array(rel)


def paired_probe(rows, kernel, T, key):
    a = {(r['task'], r['seed']): r.get('probe', {}).get(key, float('nan')) for r in cell(rows, kernel, T)}
    b = {(r['task'], r['seed']): r.get('probe', {}).get(key, float('nan')) for r in cell(rows, 'single', T)}
    keys = sorted(set(a) & set(b))
    d = []
    for k in keys:
        if np.isfinite(a[k]) and np.isfinite(b[k]):
            d.append(a[k] - b[k])
    return np.array(d)


def regression(rows):
    diffs = []
    for r in rows:
        if r['kernel'] != 'single' or r['T'] != 20:
            continue
        p = V20 / 'results/B' / f"single_overlap_a0_t{r['task']}_{r['seed']}.json"
        if p.exists():
            diffs.append(abs(r['final'] - json.loads(p.read_text())['final']))
    return (max(diffs), len(diffs)) if diffs else (float('nan'), 0)


def hypotheses(rows, chain):
    H = {}
    Ts = sorted({r['T'] for r in rows})
    ok1, txt1 = False, []
    for T in (80, 160):
        best_k, best_d, best_r = None, -9, -9
        for k in KERNELS:
            if k == 'single':
                continue
            d = paired(rows, k, T)
            r = paired_ret(rows, k, T)
            if len(d) and float(d.mean()) > best_d:
                best_k, best_d, best_r = k, float(d.mean()), float(np.mean(r)) if len(r) else float('nan')
        if best_k:
            txt1.append(f"T={T}: best={best_k} Δacc={best_d*100:+.1f}pp Δret_half={(np.exp(best_r)-1)*100:+.1f}%")
            if T == 80 and (best_d >= .05 or best_r >= np.log(1.20)):
                ok1 = True
    H['H1'] = {'text': '写入时间结构有效（T=80：Δacc≥+5pp 或 retention≥+20%）：' + '；'.join(txt1), 'ok': bool(ok1)}
    # null
    null_ok = True
    for T in (80, 160):
        for k in KERNELS:
            if k == 'single':
                continue
            d = paired(rows, k, T)
            r = paired_ret(rows, k, T)
            if len(d) and (abs(float(d.mean())) > .02 or abs(float(np.mean(r))) > .10):
                null_ok = False
    H['H1null'] = {'text': 'H1-null（所有非 single：|Δacc|≤2pp 且 |Δret|≤10%）', 'ok': bool(null_ok)}
    # H2
    gaps = {}
    for T in Ts:
        best_d = max(float(paired(rows, k, T).mean()) for k in KERNELS if k != 'single' and len(paired(rows, k, T)))
        gaps[T] = best_d
    H['H2'] = {'text': 'gap(T160)≥gap(T20)（' + '、'.join(f'T{t}={gaps[t]*100:+.1f}pp' for t in Ts) + '）',
               'ok': bool(gaps.get(160, -9) >= gaps.get(20, 9))}
    # H4：||dh|| retention（ret_half，log 比）与 acc 同向
    def ret_acc_corr(Tsub):
        xs, ys = [], []
        for T in Tsub:
            for k in KERNELS:
                if k == 'single':
                    continue
                d = paired(rows, k, T)
                r = paired_ret(rows, k, T)
                if len(d) and len(r):
                    xs.append(float(np.mean(r)))
                    ys.append(float(d.mean()))
        return (float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 2 else float('nan')), len(xs)
    corr, ncorr = ret_acc_corr(Ts)
    corr_long, nlong = ret_acc_corr([t for t in Ts if t >= 80])
    # 附加：probe 相关（exploratory）
    pc = {}
    for tag, key in [('T/2', lambda T: str(T // 2)), ('T-1', lambda T: str(T - 1))]:
        px, py = [], []
        for T in Ts:
            for k in KERNELS:
                if k == 'single':
                    continue
                d = paired(rows, k, T)
                dp = paired_probe(rows, k, T, key(T))
                if len(d) and len(dp):
                    px.append(float(np.mean(dp)))
                    py.append(float(d.mean()))
        pc[tag] = float(np.corrcoef(px, py)[0, 1]) if len(px) > 2 else float('nan')
    H['H4'] = {'text': f'||dh|| ret_half 与 acc 同向：全网格 n={ncorr} r={corr:.2f}（T≥80 子集 n={nlong} r={corr_long:.2f}）；'
                        f'probe 相关：T/2 r={pc["T/2"]:.2f}、T−1 r={pc["T-1"]:.2f}',
               'ok': bool(corr > .3)}
    # H3 chain
    if chain:
        parts, ok3 = [], False
        for task in ('xor', 'xorsw'):
            base = [r['final'] for r in chain if r['task'] == task and r['kernel'] == 'single']
            best = None
            for k in CHAIN_KERNELS:
                if k == 'single':
                    continue
                v = [r['final'] for r in chain if r['task'] == task and r['kernel'] == k]
                if v and base:
                    d = np.mean(v) - np.mean(base)
                    if best is None or d > best[1]:
                        best = (k, float(d), float(np.mean(v)))
            if best:
                parts.append(f'{task}: best={best[0]} Δ={best[1]*100:+.1f}pp acc={best[2]:.3f}')
                if best[1] >= .05 and best[2] >= .90:
                    ok3 = True
        H['H3'] = {'text': 'Chain-select T=80 最优 kernel ≥ single+5pp 且 ≥0.90：' + '；'.join(parts), 'ok': bool(ok3)}
    return H, gaps


def report(rows, chain, H, gaps):
    A = []
    ap = A.append
    ap('# 第二十四轮：Input Write Dynamics（等能量写入时间结构）')
    ap('')
    ap("协议：z_t=Σa_k B x_{t-k}（因果 FIR，Σa²=1；single=[1]、burst3/5、decay-fast/slow）；"
       "Phase A 用 V20 fixed-disjoint 固定 B、延迟任务 T{20,40,80,160}、65 epochs、5 seeds × 4 tasks；"
       "Phase B 用 V23 Chain-select T=80（learnable B hybrid / core K=5）、500 epochs。")
    ap('')
    ap('## 0. 结论摘要')
    ap('')
    ap('1. **等能量下的写入时间结构确实有作用，但只在长 T 才显著（延迟任务）**：T=20/40 无差异；'
       'T=80 最优 kernel 仅 +3.5pp（burst5，17/20 同向）；T=160 +5.3pp（burst5，16/20）。严格预注册的 H1@T=80 未通过。')
    ap('2. **优势随 T 单调增长**（H2 通过）：gap = −0.0 / +1.1 / +3.5 / +5.3pp（T=20/40/80/160），不符合“窗口对齐假象”。')
    ap('3. **在 V23 Chain-select 上写入结构是关键（H3 通过）**：T=80、core 仍为 K=5、B 仍为 hybrid 全信用时，'
       '`single` 0.777（xor）/0.913（xorsw）→ `burst5` **0.971** / `decay-slow` **0.999**；'
       '且把 single 的崩溃 seed 全部救回（xor 5/5 可达、xorsw 5/5）。**V23 的 soft-selection 失败很大程度源于单次瞬时写入。**')
    ap('4. **偏好 kernel 依赖任务**（xor→burst5、xorsw→decay-slow），说明这是“写入时间策略”的自由度，'
       '支持下一步 V24B 的可学习 write gate / scheduler（而不是固定 kernel）。')
    ap('5. **机制不是幅度也不是中程可解码性**：||Δh|| retention 与 acc 不同向（全网格 r=0.08；T≥80 子集 r=0.42）；'
       'probe 在 T/2 与 acc 负相关（r=−0.47）、在 T−1 弱正相关（r=+0.46）——'
       'kernel 改善的是“末端状态几何/可读出性”，不是简单地把扰动留得更久。')
    ap('')
    for T in sorted({r['T'] for r in rows}):
        base = mean_se([r['final'] for r in cell(rows, 'single', T)])[0]
        best_k, best_m = None, -1
        for k in KERNELS:
            if k == 'single':
                continue
            m = mean_se([r['final'] for r in cell(rows, k, T)])[0]
            if m > best_m:
                best_k, best_m = k, m
        ap(f"- Phase A T={T}: single={base:.3f}，最优 kernel {best_k}={best_m:.3f}（Δ={(best_m-base)*100:+.1f}pp）")
    reg = regression(rows)
    ap(f"- Phase 0 回归：single@T20 vs V20 fixed-disjoint singles max|Δfinal| = {reg[0]:.3e}（{reg[1]} 条配对，应为 0）")
    ap('')
    ap('## 1. Phase A：acc 汇总（4 tasks × 5 seeds）')
    ap('')
    ap('| T | ' + ' | '.join(KERNELS) + ' |')
    ap('|---|' + '---:|' * len(KERNELS))
    for T in sorted({r['T'] for r in rows}):
        cells = []
        for k in KERNELS:
            v = [r['final'] for r in cell(rows, k, T)]
            m, se = mean_se(v)
            cells.append('—' if not v else f'{m:.3f}±{se:.3f}')
        ap(f'| {T} | ' + ' | '.join(cells) + ' |')
    ap('')
    ap('| T | kernel | Δacc vs single (pp) | 同向 | ret_half 几何均值比 |')
    ap('|---|---:|---:|---:|---:|')
    for T in sorted({r['T'] for r in rows}):
        for k in KERNELS:
            if k == 'single':
                continue
            d = paired(rows, k, T)
            r = paired_ret(rows, k, T)
            if len(d):
                ap(f"| {T} | {k} | {d.mean()*100:+.2f} | {int((d>0).sum())}/{len(d)} | "
                   f"{(np.exp(np.mean(r))-1)*100:+.1f}% |")
    ap('')
    ap('## 2. Retention（t=0.5T / T−1，相对 t=5；衰减斜率）')
    ap('')
    ap('| T | kernel | ret_half | ret_end | decay_slope | lifetime |')
    ap('|---|---:|---:|---:|---:|---:|')
    for T in sorted({r['T'] for r in rows}):
        for k in KERNELS:
            rs = cell(rows, k, T)
            if not rs:
                continue
            rh = np.nanmean([r['retention']['ret_half'] for r in rs])
            re = np.nanmean([r['retention']['ret_end'] for r in rs])
            sl = np.nanmean([r['retention']['decay_slope'] for r in rs])
            lt = np.nanmean([r['retention']['lifetime'] for r in rs if np.isfinite(r['retention']['lifetime'])] or [np.nan])
            ap(f'| {T} | {k} | {rh:.3f} | {re:.3f} | {sl:+.4f} | {lt:.1f} |')
    ap('')
    if any('probe' in r for r in rows):
        ap('## 2.5 状态可解码性 probe（线性 probe，前 512 拟合 / 后 512 评估，两输出平均）')
        ap('')
        ap('| T | kernel | t=5 | t=T/2 | t=T−1 |')
        ap('|---|---:|---:|---:|---:|')
        for T in sorted({r['T'] for r in rows}):
            times = sorted({int(k) for r in cell(rows, 'single', T) for k in r.get('probe', {})})
            if not times:
                continue
            for k in KERNELS:
                rs = cell(rows, k, T)
                if not rs:
                    continue
                vals = [np.nanmean([r['probe'].get(str(t), np.nan) for r in rs]) for t in times]
                ap(f'| {T} | {k} | ' + ' | '.join(f'{v:.3f}' for v in vals) + ' |')
        ap('')
    if chain:
        ap('## 3. Phase B：Chain-select T=80（K=5, learnable B hybrid）')
        ap('')
        ap('| task | kernel | final | best-val | reach(≥0.9) |')
        ap('|---|---:|---:|---:|---:|')
        for task in ('xor', 'xorsw'):
            for k in CHAIN_KERNELS:
                rs = [r for r in chain if r['task'] == task and r['kernel'] == k]
                if rs:
                    ap(f"| {task} | {k} | {np.mean([r['final'] for r in rs]):.3f} | "
                       f"{np.mean([r['best_val'] for r in rs]):.3f} | "
                       f"{sum(1 for r in rs if r['best_val']>=.9)}/{len(rs)} |")
        ap('')
    ap('## 4. 预注册判定')
    ap('')
    for k, v in H.items():
        verdict = '**通过**' if v['ok'] is True else ('未通过' if v['ok'] is False else 'info')
        ap(f"- **{k}**：{v['text']} → {verdict}")
    ap('')
    ap('## 5. 边界')
    ap('')
    ap('- Phase A 固定 B（V20 fixed-disjoint）与核心训练协议；kernel 固定不训练，验证时间结构自由度本身。')
    ap('- Σa²=1 等总能量；无幅度扫描。')
    ap('- Phase B 的 single 是 V23 同 cell 回归锚点；结论只针对该任务族。')
    (ROOT / 'REPORT.md').write_text('\n'.join(A) + '\n')


def main():
    t0 = time.perf_counter()
    rows = delay_rows()
    chain = chain_rows()
    H, gaps = hypotheses(rows, chain)
    report(rows, chain, H, gaps)
    (ROOT / 'results/summary.json').write_text(json.dumps({'H': H, 'gaps': gaps}, indent=2, default=str))
    (ROOT / 'verification.txt').write_text('\n'.join(
        f"{k}: {'PASS' if v['ok'] else 'FAIL'} — {v['text']}" for k, v in H.items()) + '\n')
    print('analyze done in', round(time.perf_counter() - t0, 1), 's; delay', len(rows), 'chain', len(chain))
    print(json.dumps({k: v['ok'] for k, v in H.items()}, indent=1))
    for k, v in H.items():
        print(k, '::', v['text'])


if __name__ == '__main__':
    main()

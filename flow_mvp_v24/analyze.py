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


def sched_rows():
    return [r for r in load_dir('results/sched_chain') if r.get('phase') == 'sched']

def bank_chain_rows():
    return [r for r in load_dir('results/bank_chain') if r.get('phase') == 'bank']


def bank_delay_rows():
    return [r for r in load_dir('results/bank_delay') if r.get('phase') == 'bank_delay']


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


def hypotheses(rows, chain, sched=()):
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
    # H3 chain（只取 T=80 主网格）
    chain80 = [r for r in chain if r['T'] == 80]
    if chain80:
        parts, ok3 = [], False
        for task in ('xor', 'xorsw'):
            base = [r['final'] for r in chain80 if r['task'] == task and r['kernel'] == 'single']
            best = None
            for k in CHAIN_KERNELS:
                if k == 'single':
                    continue
                v = [r['final'] for r in chain80 if r['task'] == task and r['kernel'] == k]
                if v and base:
                    d = np.mean(v) - np.mean(base)
                    if best is None or d > best[1]:
                        best = (k, float(d), float(np.mean(v)))
            if best:
                parts.append(f'{task}: best={best[0]} Δ={best[1]*100:+.1f}pp acc={best[2]:.3f}')
                if best[1] >= .05 and best[2] >= .90:
                    ok3 = True
        H['H3'] = {'text': 'Chain-select T=80 最优 kernel ≥ single+5pp 且 ≥0.90：' + '；'.join(parts), 'ok': bool(ok3)}
    # ---------- Phase C（V24B）：可学习 write scheduler ----------
    if sched:
        # H5：同一 scheduler 配置在 T=80 两任务都 ≥0.95 且相对 fixed single ≥+5pp
        base_single = {t: [r['final'] for r in chain80 if r['task'] == t and r['kernel'] == 'single']
                       for t in ('xor', 'xorsw')}
        best5 = None
        for k in sorted({r['kernel'] for r in sched}):
            means = {t: float(np.mean([r['final'] for r in sched if r['kernel'] == k and r['task'] == t]))
                     for t in ('xor', 'xorsw')}
            if not all(means.values()):
                continue
            score = min(means.values())
            if best5 is None or score > best5[1]:
                best5 = (k, score, means)
        if best5:
            d_xor = best5[2]['xor'] - np.mean(base_single['xor'])
            d_xsw = best5[2]['xorsw'] - np.mean(base_single['xorsw'])
            H['H5'] = {'text': f"T=80 同一 scheduler（kernel={best5[0]}）：xor={best5[2]['xor']:.3f}"
                                f"（Δ{d_xor*100:+.1f}pp）、xorsw={best5[2]['xorsw']:.3f}（Δ{d_xsw*100:+.1f}pp），"
                                f"两任务 ≥0.95 且 Δ≥+5pp",
                       'ok': bool(best5[1] >= .95 and d_xor >= .05 and d_xsw >= .05)}
        else:
            H['H5'] = {'text': 'scheduler 数据不足', 'ok': False}
        # H6：distractor 抑制
        cells, ok_cells = [], 0
        for task in ('xor', 'xorsw'):
            for k in sorted({r['kernel'] for r in sched}):
                rs = [r for r in sched if r['kernel'] == k and r['task'] == task and r['T'] == 80]
                if not rs:
                    continue
                good = 0
                for r in rs:
                    p = r['w_profile']
                    imp = min(p['A'], p['c1'], p['B'], p['c2'])
                    if p['distractor'] <= imp - .20:
                        good += 1
                mean_all = float(np.mean([r['w_profile']['all'] for r in rs]))
                cells.append(f"{task}/{k}: {good}/{len(rs)} seeds, w_all={mean_all:.2f}")
                if good >= max(1, int(round(.8 * len(rs)))) and .02 < mean_all < 1.98:
                    ok_cells += 1
        H['H6'] = {'text': 'distractor 抑制（≥2 cells 满足 E[w|d]≤E[w|imp]−0.2）：' + '；'.join(cells),
                   'ok': bool(ok_cells >= 2)}
        # H7：T=160
        s160 = [r for r in sched if r['T'] == 160 and r['task'] == 'xor' and r['kernel'] == 'burst5']
        f160 = [r for r in chain if r['T'] == 160 and r['task'] == 'xor' and r['kernel'] == 'burst5']
        s160_single = [r for r in sched if r['T'] == 160 and r['task'] == 'xor' and r['kernel'] == 'single']
        f160_single = [r for r in chain if r['T'] == 160 and r['task'] == 'xor' and r['kernel'] == 'single']
        if s160 and f160:
            ms = float(np.mean([r['final'] for r in s160]))
            mf = float(np.mean([r['final'] for r in f160]))
            base160 = f160_single or s160_single
            mb = float(np.mean([r['final'] for r in base160])) if base160 else float('nan')
            H['H7'] = {'text': f'T=160 xor：burst5+sched={ms:.3f} vs burst5 fixed={mf:.3f}'
                                f"（需≥−2pp）、vs single={mb:.3f}（需≥+5pp）",
                       'ok': bool(ms >= mf - .02 and np.isfinite(mb) and ms >= mb + .05)}
        else:
            H['H7'] = {'text': 'T=160 scheduler 数据不足', 'ok': False}
    return H, gaps


BANK_EVENTS = ('A', 'c1', 'B', 'c2')


def bank_hypotheses(bank_chain, bank_delay, chain, rows):
    HC = {}
    chain80 = [r for r in chain if r['T'] == 80]
    # HC1 性能
    if bank_chain:
        txt, ok = [], True
        for task in ('xor', 'xorsw'):
            ad = [r['final'] for r in bank_chain if r['task'] == task]
            fixed = {k: np.mean([r['final'] for r in chain80 if r['task'] == task and r['kernel'] == k])
                     for k in CHAIN_KERNELS}
            best = max(v for v in fixed.values() if np.isfinite(v))
            m = float(np.mean(ad))
            ok &= (m >= best - .02)
            txt.append(f'{task}: adaptive={m:.3f} vs best fixed={best:.3f}（需≥−2pp）')
        HC['HC1'] = {'text': 'Chain T=80 性能：' + '；'.join(txt), 'ok': bool(ok)}
    # HC2 时间选择
    if bank_chain:
        cells, npass = [], 0
        for task in ('xor', 'xorsw'):
            rs = [r for r in bank_chain if r['task'] == task]
            dimp, dd, das = [], [], []
            for r in rs:
                p = r['profile']
                li = float(np.mean([p[e]['L_eff'] for e in BANK_EVENTS]))
                ld = p['distractor']['L_eff']
                imp_alpha = np.mean([p[e]['alpha'] for e in BANK_EVENTS], axis=0)
                das.append(float(np.abs(imp_alpha - np.array(p['distractor']['alpha'])).sum()))
                dimp.append(li)
                dd.append(ld)
            ratio = float(np.mean(dimp) / (np.mean(dd) + 1e-12))
            npos = int(sum(a > b * 1.2 for a, b in zip(dimp, dd)))
            dalpha = float(np.mean(das))
            cond = ratio >= 1.2 and npos >= max(1, int(round(.8 * len(rs)))) and dalpha > .2
            npass += int(cond)
            cells.append(f'{task}: L_imp/L_d={ratio:.2f}（{npos}/{len(rs)} seeds>1.2）、‖Δα‖₁={dalpha:.2f}')
        HC['HC2'] = {'text': '时间选择（≥1 任务：L_imp≥1.2L_d 且 ≥4/5 seeds 且 ‖Δα‖₁>0.2）：' + '；'.join(cells),
                     'ok': bool(npass >= 1)}
    # HC3 长 T
    if bank_delay:
        cells, ok = [], True
        for T in (40, 80, 160):
            ad = [r['final'] for r in bank_delay if r['T'] == T and r['task'] == 0]
            fixed = {k: np.mean([r['final'] for r in rows if r['T'] == T and r['task'] == 0
                                 and r['kernel'] == k]) for k in KERNELS}
            best = max(v for v in fixed.values() if np.isfinite(v))
            m = float(np.mean(ad)) if ad else float('nan')
            if T == 160:
                ok = np.isfinite(m) and m >= best - .02
            cells.append(f'T{T}/task0: adaptive={m:.3f} vs best fixed={best:.3f}（{len(ad)} seeds）')
        HC['HC3'] = {'text': '延迟任务 task0 长 T（T=160 需≥best fixed−2pp；T160 只跑 task0）：' + '；'.join(cells),
                     'ok': bool(ok)}
    # 结果分类
    a = HC.get('HC1', {}).get('ok'); b = HC.get('HC2', {}).get('ok')
    if a is not None and b is not None:
        verdict = ('A：自适应时间压缩成立 → V24D 连续 λ' if a and b else
                   'B：只学到更好的固定 kernel → 停止加复杂 scheduler' if a and not b else
                   'C：时间选择学到但任务不依赖 → 转 write budget' if b and not a else
                   'D：都不过 → 输入 scheduler 线暂停，转 Read Dynamics（V25）')
        HC['HC-verdict'] = {'text': f'结果分类 {verdict}（HC1={a}, HC2={b}, HC3={HC.get("HC3", {}).get("ok")}）',
                            'ok': None}
    return HC


def bank_section(ap, bank_chain, bank_delay, chain80, rows):
    if not bank_chain and not bank_delay:
        return
    ap('## 3.6 Phase D（V24C）：Adaptive Temporal Compression（kernel bank selector）')
    ap('')
    ap('结构：`α_t=softmax(MLP(x_t,meta_t))`（9→16→5，bias 初始 [4,0,0,0,0]）；`ã_t=α_tK/‖α_tK‖`（Σa²=1）；'
       '`z_{t+τ} += ã_{t,τ}B x_t`；无 w_t；core 仍 K=5、B/scheduler 走 hybrid full。')
    ap('')
    if bank_chain:
        ap('### Chain-select T=80：adaptive vs fixed')
        ap('')
        ap('| task | adaptive final | best | reach | fixed best（kernel） | probe T/2 | probe T−1 |')
        ap('|---|---:|---:|---:|---|---:|---:|')
        for task in ('xor', 'xorsw'):
            rs = [r for r in bank_chain if r['task'] == task]
            if not rs:
                continue
            fixed = {k: np.mean([r['final'] for r in chain80 if r['task'] == task and r['kernel'] == k])
                     for k in CHAIN_KERNELS}
            bk = max(fixed, key=lambda k: fixed[k])
            pr = {t: np.mean([r.get('probe', {}).get(str(t), np.nan) for r in rs]) for t in (40, 79)}
            ap(f"| {task} | {np.mean([r['final'] for r in rs]):.3f} | {fixed[bk]:.3f} | "
               f"{sum(1 for r in rs if r['best_val']>=.9)}/{len(rs)} | {bk} | {pr[40]:.3f} | {pr[79]:.3f} |")
        ap('')
        ap('### 事件级 α 与有效时间长度 `L_eff=Σ τ ã_τ²`')
        ap('')
        ap('| task | 事件 | α(single,burst3,burst5,dfast,dslow) | L_eff |')
        ap('|---|---|---|---:|')
        for task in ('xor', 'xorsw'):
            rs = [r for r in bank_chain if r['task'] == task]
            if not rs:
                continue
            for e in ('A', 'c1', 'distractor', 'B', 'c2'):
                al = np.mean([r['profile'][e]['alpha'] for r in rs], axis=0)
                le = np.mean([r['profile'][e]['L_eff'] for r in rs])
                ap(f"| {task} | {e} | " + ','.join(f'{v:.2f}' for v in al) + f" | {le:.2f} |")
        ap('')
    if bank_delay:
        ap('### 延迟任务：adaptive vs fixed（含 probe t=T/2 / T−1）')
        ap('')
        ap('| T | adaptive final | best fixed（kernel） | single | adaptive probe T/2 | adaptive probe T−1 | single probe T−1 |（全部 task0）')
        ap('|---|---:|---|---:|---:|---:|---:|')
        for T in (40, 80, 160):
            ad = [r for r in bank_delay if r['T'] == T and r['task'] == 0]
            if not ad:
                continue
            fixed = {k: np.mean([r['final'] for r in rows if r['T'] == T and r['task'] == 0
                                 and r['kernel'] == k]) for k in KERNELS}
            bk = max(fixed, key=lambda k: fixed[k])
            m = np.mean([r['final'] for r in ad])
            p_half = np.nanmean([r.get('probe', {}).get(str(T // 2), np.nan) for r in ad])
            p_end = np.nanmean([r.get('probe', {}).get(str(T - 1), np.nan) for r in ad])
            s_end = np.nanmean([r.get('probe', {}).get(str(T - 1), np.nan)
                                for r in rows if r['T'] == T and r['task'] == 0 and r['kernel'] == 'single'])
            ap(f"| {T} | {m:.3f} | {fixed[bk]:.3f}（{bk}） | {fixed['single']:.3f} | {p_half:.3f} | {p_end:.3f} | {s_end:.3f} |")
        ap('')
        ap('| T | 事件 | α(single,burst3,burst5,dfast,dslow) | L_eff |')
        ap('|---|---|---|---:|')
        for T in (40, 80, 160):
            ad = [r for r in bank_delay if r['T'] == T and r['task'] == 0]
            if not ad:
                continue
            for e in ('A', 'B'):
                al = np.mean([r['profile'][e]['alpha'] for r in ad], axis=0)
                le = np.mean([r['profile'][e]['L_eff'] for r in ad])
                ap(f"| {T} | {e} | " + ','.join(f'{v:.2f}' for v in al) + f" | {le:.2f} |")
        ap('')


def report(rows, chain, sched, bank_chain, bank_delay, H, gaps):
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
                vals = [np.nanmean([r.get('probe', {}).get(str(t), np.nan) for r in rs]) for t in times]
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
    if sched:
        ap('## 3.5 Phase C（V24B）：可学习 Write Scheduler `w_t`')
        ap('')
        ap('结构：`w_t=2σ(MLP(x_t,meta_t))`（不看 h），`e_t=w_t⊙x_t`，`z_t=Σ_k a_k(e_{t-k}B)`；'
           'controller 末层零初始化 → 初始 w≡1（与固定 kernel 起点逐值一致）；w 与 B 走 full BPTT（hybrid），core 仍 K=5。')
        ap('')
        ap('### Q1：能力（T=80，xor/xorsw）')
        ap('')
        ap('| task | kernel | fixed final | sched final | Δ | fixed best-val | sched best-val | sched reach |')
        ap('|---|---:|---:|---:|---:|---:|---:|---:|')
        for task in ('xor', 'xorsw'):
            for k in sorted({r['kernel'] for r in sched}):
                fr = [r for r in chain if r['task'] == task and r['kernel'] == k and r['T'] == 80]
                sr = [r for r in sched if r['task'] == task and r['kernel'] == k and r['T'] == 80]
                if not sr:
                    continue
                sm = np.mean([r['final'] for r in sr])
                fm = np.mean([r['final'] for r in fr]) if fr else float('nan')
                sb = np.mean([r['best_val'] for r in sr])
                fb = np.mean([r['best_val'] for r in fr]) if fr else float('nan')
                ap(f"| {task} | {k} | {fm:.3f} | {sm:.3f} | {(sm-fm)*100:+.1f}pp | {fb:.3f} | {sb:.3f} | "
                   f"{sum(1 for r in sr if r['best_val']>=.9)}/{len(sr)} |")
        ap('')
        ap('### Q2：事件级 w（均值；distractor 应最小）')
        ap('')
        ap('| task | kernel | A | c1 | distractor | B | c2 | all |')
        ap('|---|---:|---:|---:|---:|---:|---:|---:|')
        for task in ('xor', 'xorsw'):
            for k in sorted({r['kernel'] for r in sched}):
                rs = [r for r in sched if r['task'] == task and r['kernel'] == k and r['T'] == 80]
                if not rs:
                    continue
                prof = {key: np.mean([r['w_profile'][key] for r in rs])
                        for key in ('A', 'c1', 'distractor', 'B', 'c2', 'all')}
                ap(f"| {task} | {k} | {prof['A']:.2f} | {prof['c1']:.2f} | **{prof['distractor']:.2f}** | "
                   f"{prof['B']:.2f} | {prof['c2']:.2f} | {prof['all']:.2f} |")
        ap('')
        s160 = [r for r in sched if r['T'] == 160]
        if s160:
            ap('### Q3：T=160 xor')
            ap('')
            ap('| arm | final | best-val | reach |')
            ap('|---|---:|---:|---:|')
            for tag, rs in [('single fixed', [r for r in chain if r['T'] == 160 and r['task'] == 'xor' and r['kernel'] == 'single']),
                            ('burst5 fixed', [r for r in chain if r['T'] == 160 and r['task'] == 'xor' and r['kernel'] == 'burst5']),
                            ('burst5+sched', [r for r in s160 if r['task'] == 'xor' and r['kernel'] == 'burst5'])]:
                if rs:
                    ap(f"| {tag} | {np.mean([r['final'] for r in rs]):.3f} | {np.mean([r['best_val'] for r in rs]):.3f} | "
                       f"{sum(1 for r in rs if r['best_val']>=.9)}/{len(rs)} |")
            ap('')
    bank_section(ap, bank_chain, bank_delay, [r for r in chain if r['T'] == 80], rows)
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
    ap('- Phase C 中 controller 只看 `(x_t, meta_t)`（不看 h）、第一版固定 `a_k`；w 与 B 同走 hybrid full 通道。')
    if sched:
        ap('')
        ap('### Phase C 解释（exploratory）')
        ap('')
        ap('- **能力（Q1）**：同一 scheduler 配置没有超过"每任务人工选最优固定 kernel"：xor 最好的 fixed 是 burst5（0.971），'
           '最好的 sched 是 decay-slow（0.946）；xorsw 最好的 fixed/sched 都是 decay-slow（0.999）。sched 的方差更大'
           '（burst5+sched xor 有一条 0.733 的崩溃；single+sched 既救回 0.527 也弄坏 1.0 的 seed）。')
        ap('- **重要性（Q2）**：30 条流里没有一条学会相对抑制 distractor（H6 0/30）。更糟的是常见两种退化：'
           '（a）**全开放大**：xor/burst5、xor/decay-slow 的 w_all≈1.45/1.56，所有事件（含 distractor）都被写到 >1；'
           '（b）**全局压缩 + 末端保留**：xorsw 的各臂把 A/c1/distractor/B 压到 0.4–0.8，只把 c2 留下 ~1.7。'
           '两者都不是"重要多写、噪声少写"。')
        ap('- **为什么标量 input gate 很容易退化的机制解释**：任务是符号分类，`y=sign(v)`，对输入做**正标量缩放不改变可解性**'
           '（网络可用内部增益补偿），因此"写多强"的最优点是一大片平台；梯度把 `w` 推向放大（更小有效噪声）或整体压缩，'
           '而不是相对选择性。相对重要性只有在"写重了会挤占别的信息"的预算约束下才有梯度——当前的加法写入没有这种竞争。')
        ap('- **长 T（Q3）**：T=160 的 Chain-select 连 fixed `burst5` 都只有 0.777（reach 2/3），scheduler 0.743（reach 0/3）没有展示空间；'
           'V24A 的长 T 写入结构收益只在延迟任务上成立，不能外推到 Chain-select。')
        ap('- **结论**：V24B 第一版（标量 `w_t`）失败：它既没有稳定达到最优固定 kernel，也没有学会重要性。'
           '下一步优先做 **V24C：动态 `λ_t`（trace shaping / 记多久）**，或给 scheduler 加**写入预算约束**（迫使事件竞争），'
           '而不是继续调 `w_t` 的容量。')
    (ROOT / 'REPORT.md').write_text('\n'.join(A) + '\n')


def main():
    t0 = time.perf_counter()
    rows = delay_rows()
    chain = chain_rows()
    sched = sched_rows()
    bank_chain = bank_chain_rows()
    bank_delay = bank_delay_rows()
    H, gaps = hypotheses(rows, chain, sched)
    H.update(bank_hypotheses(bank_chain, bank_delay, chain, rows))
    report(rows, chain, sched, bank_chain, bank_delay, H, gaps)
    (ROOT / 'results/summary.json').write_text(json.dumps({'H': H, 'gaps': gaps}, indent=2, default=str))
    (ROOT / 'verification.txt').write_text('\n'.join(
        f"{k}: {'PASS' if v['ok'] is True else ('FAIL' if v['ok'] is False else 'info')} — {v['text']}"
        for k, v in H.items()) + '\n')
    print('analyze done in', round(time.perf_counter() - t0, 1), 's; delay', len(rows), 'chain', len(chain))
    print(json.dumps({k: v['ok'] for k, v in H.items()}, indent=1))
    for k, v in H.items():
        print(k, '::', v['text'])


if __name__ == '__main__':
    main()

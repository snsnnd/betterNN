"""第二十三轮汇总：Chain-select 的 K×T 网格、breakpoint 判定、REPORT.md。"""
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
DIRS = ['results/main', 'results/control', 'results/scale']
PRIMARY_VARIANT = 'transient'


def load_rows():
    rows = []
    for d in DIRS:
        p = ROOT / d
        if p.exists():
            for f in sorted(p.glob('*.json')):
                rows.append(json.loads(f.read_text()))
    return rows


def load_rescue():
    p = ROOT / 'results/rescue_probe.json'
    return json.loads(p.read_text()) if p.exists() else []


def cell(rows, variant, task, T, K):
    return [r for r in rows if r['variant'] == variant and r['task'] == task
            and r['T'] == T and r['K'] == str(K)]


def best_val(r):
    curve = r.get('curve') or []
    return max(c['acc'] for c in curve) if curve else float('nan')


def mean_se(vals):
    vals = np.asarray(vals, float)
    if not len(vals):
        return float('nan'), float('nan')
    se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return float(vals.mean()), float(se)


def paired(rows, variant, task, T, Ka, Kb, key='final'):
    a = {r['seed']: r[key] for r in cell(rows, variant, task, T, Ka)}
    b = {r['seed']: r[key] for r in cell(rows, variant, task, T, Kb)}
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds])
    return seeds, d


def fmt(x):
    return '—' if (isinstance(x, float) and not np.isfinite(x)) else f'{x:.3f}'


def report(rows):
    A = []
    ap = A.append
    ap('# 第二十三轮：Credit-Stress Benchmark（长程 core 信用压力任务）')
    ap('')
    ap("协议：Chain-select（A@.05T / c1@.21T / distractor@.39T / B@.58T / c2@.76T，全部瞬态脉冲；读出在最后一步）；"
       "B 恒可训练恒 Full；core {W,route,hold} 截断窗口 K∈{5,10,20,Full}；500 epochs、5 seeds（T=160 为 3 seeds）；"
       "指标 = 两输出全对率（chance≈0.25）。")
    ap('')
    ap('## 0. 结论摘要')
    ap('')
    for line in summary_lines(rows):
        ap(line)
    rescue = load_rescue()
    if rescue:
        ap(f"- 救援诊断（T=20 xor、K=5 失败流续训 Full 100 epochs）："
           f"mean K5={np.mean([r['K5']['acc'] for r in rescue]):.3f} → rescue={np.mean([r['rescue']['acc'] for r in rescue]):.3f}"
           f"（0/5 被救回）——失败是吸引盆，不是缺长程信用。")
    ap('- **总判定**：没有找到"5 步系统失败、Full 系统成功"的长程硬案例；T≥40 时 K=5 的可达性 ≥ Full（T=80 xor：'
       'K5 4/5 vs Full 3/5；T=160：K5 2/3 vs Full 2/3）；差异由训练"软选择平台/振荡"主导，Full 并不更稳。')
    ap('')
    ap('## 1. 主网格（transient）')
    ap('')
    for task in ('xor', 'xorsw'):
        ap(f'### {task}')
        ap('')
        ap('| T | K=5 | K=10 | K=20 | Full | Full−K5 (pp) | 同向 | best-val reach(Full/K5) |')
        ap('|---|---:|---:|---:|---:|---:|---:|---|')
        for T in sorted({r['T'] for r in rows if r['variant'] == PRIMARY_VARIANT and r['task'] == task}):
            cells = {}
            for K in ('5', '10', '20', 'full'):
                rs = cell(rows, PRIMARY_VARIANT, task, T, K)
                m, se = mean_se([r['final'] for r in rs])
                cells[K] = (m, se, len(rs))
            seeds, d = paired(rows, PRIMARY_VARIANT, task, T, 'full', '5')
            gap = float(d.mean() * 100) if len(d) else float('nan')
            npos = int((d > 0).sum())
            reach = []
            for K in ('full', '5'):
                rs = cell(rows, PRIMARY_VARIANT, task, T, K)
                reach.append(f"{sum(1 for r in rs if best_val(r) >= .9)}/{len(rs)}")
            row = [str(T)]
            for K in ('5', '10', '20', 'full'):
                m, se, n = cells[K]
                row.append('—' if not n else f'{m:.3f}±{se:.3f}')
            row.append(f'{gap:+.1f}' if np.isfinite(gap) else '—')
            row.append(f'{npos}/{len(d)}' if len(d) else '—')
            row.append(f'{reach[0]} / {reach[1]}')
            ap('| ' + ' | '.join(row) + ' |')
        ap('')
    ap('## 2. 对照（sustained，T=80）')
    ap('')
    ap('| task | K=5 | K=20 | Full | Full−K5 (pp) | best-val reach(Full/K5) |')
    ap('|---|---:|---:|---:|---:|---|')
    for task in ('xor', 'xorsw'):
        cells = {}
        for K in ('5', '20', 'full'):
            rs = cell(rows, 'sustained', task, 80, K)
            cells[K] = mean_se([r['final'] for r in rs])[0] if rs else float('nan')
        _, d = paired(rows, 'sustained', task, 80, 'full', '5')
        gap = float(d.mean() * 100) if len(d) else float('nan')
        reach = []
        for K in ('full', '5'):
            rs = cell(rows, 'sustained', task, 80, K)
            reach.append(f"{sum(1 for r in rs if best_val(r) >= .9)}/{len(rs)}")
        ap(f"| {task} | {fmt(cells['5'])} | {fmt(cells['20'])} | {fmt(cells['full'])} | "
           f"{gap:+.1f} | {reach[0]} / {reach[1]} |" if np.isfinite(gap) else
           f"| {task} | {fmt(cells['5'])} | {fmt(cells['20'])} | {fmt(cells['full'])} | — | — |")
    ap('')
    ap('## 3. 预注册判定')
    ap('')
    H = hypotheses(rows)
    for k, v in H.items():
        verdict = '**通过**' if v['ok'] is True else ('未通过' if v['ok'] is False else 'info')
        ap(f"- **{k}**：{v['text']} → {verdict}")
    ap('')
    ap('## 4. 救援诊断（T=20 xor K=5 失败流 → Full 微调）')
    ap('')
    if rescue:
        ap('| seed | K=5 acc | +Full 100ep | K=5 y0 | rescue y0 |')
        ap('|---|---:|---:|---:|---:|')
        for r in rescue:
            ap(f"| {r['seed']} | {r['K5']['acc']:.3f} | {r['rescue']['acc']:.3f} | "
               f"{r['K5']['y0']:.3f} | {r['rescue']['y0']:.3f} |")
        ap('')
        ap(f"- mean：K5 {np.mean([r['K5']['acc'] for r in rescue]):.3f} → rescue "
           f"{np.mean([r['rescue']['acc'] for r in rescue]):.3f}（无一条被救回）。")
        ap('- 说明 K=5 训练把模型带进一个 **吸引盆（软选择平台）**：即使后续给 core 完整 20 步信用，'
           '100 epochs 内也无法离开；因此 T=20 的 K=5 失败是**轨迹/盆地选择**，不是表示能力或信用 horizon 不足。')
    else:
        ap('（未运行）')
    ap('')
    ap('## 5. 解释（exploratory）')
    ap('')
    ap('- 没有长程需求：T≥40 时 K=5 的可达性与 Full 相同或更好；T=160 时 Full 自身也只有 2/3 可达。'
       '唯一干净的窗口效应（xor T=20 K=5 0/5 可达）是**短 T 的脉冲-边界对齐**：'
       'K=5 的末窗 [15..19] 只含 c2@15、缺 B@12，而 K=10 的 [10..19] 覆盖 B。')
    ap('- 主失败模式是"软选择平台"（y0≈0.75–0.94）与周期振荡：Full/T=40、T=80 的可达性只有 3/5；'
       '因此 final accuracy 的 K 差异不能解释为 credit horizon 差异。')
    ap('- sustained 对照（c1/c2 持续可见）：Full 5/5 可达，K=5 只有 2/5（xor）/4/5（xorsw）；'
       '而 transient 在该 T 的 K5 可达性反而更高（4/5）——说明瓶颈不是"命令是否可见"，'
       '而是选择计算的优化盆地（且 n=5，方向本身也有噪声）。')
    ap('- 对 Flow-v3 的含义：**不要为 core 设计长程信用机制**；相反，"更长的 core 信用"在 Chain-select 上'
       '并不更稳（Full 的可达性更低、崩溃更多）。真正稳定的是短窗口 + B 的 Full credit。')
    ap('')
    ap('## 6. 分解与边界')
    ap('')
    h6 = H.get('H6', {})
    ap(f"- H6（exploratory）：{h6.get('text', '—')}")
    ap('- Full 校准按固定最后 epoch（不用测试集选模）；K<Full 用 V22 双通道（core 截断 / B+readout 完整）。')
    ap('- 两个任务互为镜像；结论限于该任务族（合成二分类、±1 值、±2 context 脉冲）。')
    ap('- T=160 只跑 xor、3 seeds、同一 500 epochs（算力边界，见 PLAN §3/§6）。')
    (ROOT / 'REPORT.md').write_text('\n'.join(A) + '\n')
    return H


def hypotheses(rows):
    H = {}
    # H0 calibration
    ok0, txt0 = True, []
    for task in ('xor', 'xorsw'):
        for T in (20, 80):
            rs = cell(rows, PRIMARY_VARIANT, task, T, 'full')
            if not rs:
                ok0 = False
                txt0.append(f'{task}/T{T}=missing')
                continue
            m = float(np.mean([r['final'] for r in rs]))
            txt0.append(f'{task}/T{T}={m:.3f}')
            ok0 &= m >= .90
    H['H0'] = {'text': 'transient Full 校准（' + '、'.join(txt0) + '）≥0.90', 'ok': bool(ok0)}
    # H1 T=80 per task
    h1_txt, h1_ok = [], True
    for task in ('xor', 'xorsw'):
        seeds, d = paired(rows, PRIMARY_VARIANT, task, 80, 'full', '5')
        full = [r['final'] for r in cell(rows, PRIMARY_VARIANT, task, 80, 'full')]
        full_m = float(np.mean(full)) if full else float('nan')
        gap = float(d.mean()) if len(d) else float('nan')
        npos = int((d > 0).sum())
        cond = bool(np.isfinite(gap) and full_m >= .90 and gap >= .15 and npos >= .8 * len(d))
        h1_ok &= cond
        h1_txt.append(f'{task}: Full={full_m:.3f}, gap={gap*100:+.1f}pp, {npos}/{len(d)}')
    H['H1'] = {'text': 'T=80 breakpoint（' + '；'.join(h1_txt) + '）', 'ok': bool(h1_ok)}
    # H2 window content at T=80 (both tasks pooled? per task then require both)
    h2a, h2b = [], []
    for task in ('xor', 'xorsw'):
        _, d20 = paired(rows, PRIMARY_VARIANT, task, 80, '20', '5')
        _, d10 = paired(rows, PRIMARY_VARIANT, task, 80, '10', '5')
        h2a.append(float(d20.mean()) if len(d20) else float('nan'))
        h2b.append(float(d10.mean()) if len(d10) else float('nan'))
    H['H2'] = {'text': f"T=80: mean(K20−K5)={np.nanmean(h2a)*100:+.1f}pp ≥15 且 mean(K10−K5)={np.nanmean(h2b)*100:+.1f}pp ≤5",
               'ok': bool(np.nanmean(h2a) >= .15 and np.nanmean(h2b) <= .05)}
    # H3 T scaling xor transient
    _, d20 = paired(rows, PRIMARY_VARIANT, 'xor', 20, 'full', '5')
    _, d80 = paired(rows, PRIMARY_VARIANT, 'xor', 80, 'full', '5')
    g20 = float(d20.mean()) if len(d20) else float('nan')
    g80 = float(d80.mean()) if len(d80) else float('nan')
    H['H3'] = {'text': f"xor: gap(T80)={g80*100:+.1f}pp ≥ gap(T20)={g20*100:+.1f}pp + 10",
               'ok': bool(np.isfinite(g20) and np.isfinite(g80) and g80 >= g20 + .10)}
    # H4 sustained control at T=80
    hs_t, hs_s = [], []
    for task in ('xor', 'xorsw'):
        _, dt = paired(rows, PRIMARY_VARIANT, task, 80, 'full', '5')
        _, ds = paired(rows, 'sustained', task, 80, 'full', '5')
        hs_t.append(float(dt.mean()) if len(dt) else float('nan'))
        hs_s.append(float(ds.mean()) if len(ds) else float('nan'))
    H['H4'] = {'text': f"T=80: transient gap={np.nanmean(hs_t)*100:+.1f}pp（应≥15），sustained gap={np.nanmean(hs_s)*100:+.1f}pp（应≤5）",
               'ok': bool(np.nanmean(hs_t) >= .15 and np.nanmean(hs_s) <= .05)}
    # H5 T=160: K20 fails, Full holds
    full160 = [r['final'] for r in cell(rows, PRIMARY_VARIANT, 'xor', 160, 'full')]
    _, d5 = paired(rows, PRIMARY_VARIANT, 'xor', 160, 'full', '5')
    _, d20_ = paired(rows, PRIMARY_VARIANT, 'xor', 160, 'full', '20')
    f160 = float(np.mean(full160)) if full160 else float('nan')
    g5_160 = float(d5.mean()) if len(d5) else float('nan')
    g20_160 = float(d20_.mean()) if len(d20_) else float('nan')
    H['H5'] = {'text': f"T=160: Full={f160:.3f}≥0.90；gap(K5)={g5_160*100:+.1f}pp、gap(K20)={g20_160*100:+.1f}pp（K20 应≥15）",
               'ok': bool(np.isfinite(f160) and f160 >= .90 and np.isfinite(g20_160) and g20_160 >= .15)}
    # H6 decomposition exploratory
    parts = []
    for task in ('xor', 'xorsw'):
        rs = cell(rows, PRIMARY_VARIANT, task, 80, '5')
        if rs:
            parts.append(f"{task}: y0={np.mean([r['y0'] for r in rs]):.3f}, y1={np.mean([r['y1'] for r in rs]):.3f}")
    H['H6'] = {'text': 'T=80 K=5 的 y0/y1（' + '；'.join(parts) + '）——预期 y1 高、y0 低', 'ok': None}
    return H


def summary_lines(rows):
    lines = []
    for task in ('xor', 'xorsw'):
        for T in (20, 80):
            fr = cell(rows, PRIMARY_VARIANT, task, T, 'full')
            k5 = cell(rows, PRIMARY_VARIANT, task, T, '5')
            if fr and k5:
                fm = np.mean([r['final'] for r in fr])
                km = np.mean([r['final'] for r in k5])
                rf = sum(1 for r in fr if best_val(r) >= .9)
                r5 = sum(1 for r in k5 if best_val(r) >= .9)
                lines.append(f"- {task} T={T}: final Full={fm:.3f} vs K5={km:.3f}（Δ={(fm-km)*100:+.1f}pp）；"
                             f"可达性(best-val≥0.9) Full {rf}/{len(fr)} vs K5 {r5}/{len(k5)}")
    return lines


def main():
    t0 = time.perf_counter()
    rows = load_rows()
    H = report(rows)
    (ROOT / 'results/summary.json').write_text(json.dumps(H, indent=2, default=str))
    (ROOT / 'verification.txt').write_text('\n'.join(
        f"{k}: {'PASS' if v['ok'] else 'FAIL' if v['ok'] is not None else 'info'} — {v['text']}"
        for k, v in H.items()) + '\n')
    print('analyze done in', round(time.perf_counter() - t0, 1), 's; rows', len(rows))
    print(json.dumps({k: v['ok'] for k, v in H.items()}, indent=1))
    for k, v in H.items():
        if v['ok'] is None or v['ok'] is False:
            print(k, '::', v['text'])


if __name__ == '__main__':
    main()

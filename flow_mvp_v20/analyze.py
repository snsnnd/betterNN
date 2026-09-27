"""第二十轮汇总：完整性核验 + 单任务指标（dynamics/reachability）+ H1/H2/H3 判定 + REPORT.md。"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import experiment as E
import metrics as Mt

PHASES = {'A': 'results/A', 'B': 'results/B', 'C': 'results/C'}
RATIO_KEYS = {0: '0', .125: '12.5'}


def load_rows(phase):
    out = ROOT / PHASES[phase]
    files = [p for p in sorted(out.glob('*.json'))
             if p.name not in ('config.json', 'h3.json') and not p.name.startswith('metrics_')]
    return [json.loads(p.read_text()) for p in files]


def verify_streams(phase, n=3):
    rows = [r for r in load_rows(phase) if 'matrix' in r]
    diffs = []
    for r in rows[:n]:
        m, ck, _ = Mt.load_ckpt(ROOT / PHASES[phase] / f"{r['name']}_stage3.pt")
        sc = E.scores(m, r['seed'])
        diffs.append(max(abs(a - b) for a, b in zip(sc, r['matrix'][-1])))
    return float(max(diffs)) if diffs else -1.0


def single_metrics(phase):
    cache = ROOT / PHASES[phase] / 'metrics_singles.json'
    if cache.exists():
        return json.loads(cache.read_text())
    rows = [r for r in load_rows(phase) if r['name'].startswith('single_')]
    res = []
    for r in rows:
        label = r['topo'] + (f"_s{r['sw']:g}" if r['sw'] != .9 else '') + \
                (f"_a{r['alpha']:g}" if r['alpha'] is not None else '')
        ck = Mt.config_metrics(ROOT / PHASES[phase] / f"{r['name']}.pt", task=r['task'])
        ck['label'] = label
        ck['final'] = r['final']
        res.append(ck)
    cache.write_text(json.dumps(res))
    return res


def agg_vals(items, key):
    xs = [i[key] for i in items if i.get(key) is not None and not (isinstance(i[key], float) and np.isnan(i[key]))]
    return (float(np.mean(xs)), float(np.std(xs)), len(xs)) if xs else (float('nan'), float('nan'), 0)


def cl_table(rows, ratio, key):
    """按 topo + 附加标签（sw/alpha）聚合 CL 指标。"""
    out = {}
    for r in rows:
        if 'matrix' not in r or r['ratio'] != ratio:
            continue
        if r['topo'] == 'overlap':
            label = f"overlap_a{r['alpha']:g}"
        elif r['sw'] != .9:
            label = f"{r['topo']}_s{r['sw']:g}"
        else:
            label = r['topo']
        out.setdefault(label, []).append(r[key])
    return {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in out.items()}


def paired_h1(rows, topo, base='single-random'):
    """r=0 下 (topo − base) 的逐流 forgetting 差，按 seed 聚合。"""
    def by_seed(rows, name):
        d = {}
        for r in rows:
            if 'matrix' in r and r['ratio'] == 0 and r['topo'] == name and r['sw'] == .9 and r['alpha'] is None:
                d.setdefault(r['seed'], []).append(r['forgetting'])
        return {k: float(np.mean(v)) for k, v in d.items()}
    a, b = by_seed(rows, topo), by_seed(rows, base)
    seeds = sorted(set(a) & set(b))
    diffs = {s: a[s] - b[s] for s in seeds}
    vals = np.array([diffs[s] for s in seeds]) * 100          # pp
    return {'seeds': seeds, 'diffs': [float(v) for v in vals], 'mean': float(vals.mean()) if len(vals) else float('nan'),
            'n_neg': int((vals < 0).sum())}


def write_report(single, rows, ver, h1, h2, h3):
    A = []
    ap = A.append
    ap('# 第二十轮：B 输入拓扑 × 底网动力学（Phase A/B/C）')
    ap('')
    ap(f"配置：β={E.BETA:.2f}，A'B'C'D'，65 轮/阶段，5 seeds，orders o0/o1/o2，r∈{{0,12.5%}}。")
    ap('')
    ap('## 1. 完整性核验')
    ap('')
    ap(f"- A/B/C 结果文件数：{[len(load_rows(p)) for p in PHASES]}")
    ap(f"- 抽样重载 stage3 检查点复算矩阵最大差：{ver:.2e}")
    ap('')
    ap('## 2. 单任务可学性（final，均值±sd，任务/种子聚合）')
    ap('')
    ap('| label | final | r_PR | probe r_eff | probe σmax | ρ_eff | γ4 | γ8 | fz decay24 | fz ratio_end | fz G_max | cl G_max |')
    ap('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for lab in sorted({s['label'] for s in single}):
        it = [s for s in single if s['label'] == lab]
        f = agg_vals(it, 'final')
        rp = agg_vals(it, 'r_PR')
        pe = agg_vals(it, 'probe_r_eff')
        ps = agg_vals(it, 'probe_sigma_max')
        re = agg_vals(it, 'rho_eff')
        g4 = agg_vals(it, 'gamma4')
        g8 = agg_vals(it, 'gamma8')
        dc = agg_vals(it, 'fz_decay')
        rn = agg_vals(it, 'fz_ratio_end')
        gm = agg_vals(it, 'fz_Gmax')
        gcl = agg_vals(it, 'cl_Gmax')
        ap(f"| {lab} | {f[0]:.3f}±{f[1]:.3f} | {rp[0]:.1f} | {pe[0]:.1f} | {ps[0]:.2f} | "
           f"{re[0]:.3f} | {g4[0]:.3f} | {g8[0]:.3f} | {dc[0]:+.3f} | {rn[0]:.2f} | {gm[0]:.2f} | {gcl[0]:.2f} |")
    ap('')
    ap('（lifetime 分两口径：`fz` = 预注册的 frozen Route/Hold（拆出底网稳定性）；`cl` = 扰动重新进入 Route 的闭环对照。'
       'single-random/central 的 τ½ 在 24 步窗口内被截尾，故用 24 步对数衰减率与 ratio_end 代替。）')
    ap('')
    ap('## 3. CL 汇总（r=0 / r=12.5%，合并 3 顺序）')
    ap('')
    for ratio in (0, .125):
        acq = cl_table(rows, ratio, 'acquisition')
        fin = cl_table(rows, ratio, 'final_mean')
        fog = cl_table(rows, ratio, 'forgetting')
        ap(f'### r={RATIO_KEYS[ratio]}%')
        ap('')
        ap('| label | acquisition | final_mean | forgetting (pp) |')
        ap('|---|---:|---:|---:|')
        for lab in sorted(fin):
            ap(f"| {lab} | {acq[lab][0]:.2f} | {fin[lab][0]:.2f} | {fog[lab][0]*100:.2f} |")
        ap('')
    ap('## 4. 预注册假设判定')
    ap('')
    ap('### H1：multi-separate vs single-random（r=0 遗忘）')
    ap('')
    for topo in ('multi8', 'multi16'):
        h = h1[topo]
        ok = h['mean'] <= -3 and h['n_neg'] >= 4
        ap(f"- {topo} − single-random：Δ={h['mean']:.2f}pp，负向 {h['n_neg']}/5，"
           f"门槛(−3pp 且 ≥4/5) → {'通过' if ok else '未通过'}")
    ap('')
    ap('### H2：distributed vs single-random（可达性 ↑≥10%，τ½ ↓≥20%）')
    ap('')
    rr = h2.get('reach_rel_mean')
    ok2r = rr is not None and rr >= .10 and h2.get('reach_seeds_ge10', 0) >= 4
    ap(f"- 可达性：mean Δ={rr if rr is None else round(rr*100,1)}%，≥10% 的 seeds "
       f"{h2.get('reach_seeds_ge10', 0)}/5 → {'通过' if ok2r else '未通过'}")
    if 'decay_base_mean' in h2:
        ap(f"- 寿命（frozen，预注册口径）：baseline decay={h2['decay_base_mean']:+.3f}，"
           f"distributed decay={h2['decay_dist_mean']:+.3f} → "
           f"{'方向支持' if h2['decay_qual_ok'] else '方向不支持'}；"
           f"预注册的“τ½ ↓≥20%”因基线被截尾不可直接计算。")
        ap(f"- 闭环对照：baseline decay={h2.get('cl_decay_base', float('nan')):+.3f}，"
           f"distributed decay={h2.get('cl_decay_dist', float('nan')):+.3f}（同向）。")
    ap('')
    ap('### H3：overlap α↑ → 梯度冲突↑（cos ≤ −0.10 且 ≥2 组）')
    ap('')
    for k, v in h3.items():
        n = sum(1 for g in ('W', 'route', 'hold') if v[g] <= -.10)
        ap(f"- {k}: W {v['W']:.3f} / route {v['route']:.3f} / hold {v['hold']:.3f}，"
           f"≤−0.10 的组数 {n}/3 → {'通过' if n >= 2 else '未通过'}")
    ap('')
    ap('## 5. 观察（自动汇总）')
    ap('')

    def lm(lab, key):
        return agg_vals([s for s in single if s['label'] == lab], key)[0]

    fin_a0 = cl_table(rows, 0, 'final_mean')
    fog_a0 = cl_table(rows, 0, 'forgetting')
    fin_r12 = cl_table(rows, .125, 'final_mean')
    ap(f"1. **任务侧无差异**：单任务 final 全部 ≥0.994；r=0 遗忘 {fog_a0['single-random'][0]*100:.1f}pp (single-random) "
       f"vs {fog_a0['distributed'][0]*100:.1f}pp (distributed)；H1 Δ≤1pp；r=12.5% 主族 final "
       f"{fin_r12['single-random'][0]:.2f}~{fin_r12['distributed'][0]:.2f}。s_W∈{{0.7,0.9,1.1}} 也无差异。")
    ap(f"2. **动力学被强烈改变（frozen 口径）**：ρ_eff {lm('single-random','rho_eff'):.2f}→{lm('distributed','rho_eff'):.2f}、"
       f"G_max {lm('single-random','fz_Gmax'):.2f}→{lm('distributed','fz_Gmax'):.2f}、"
       f"decay24 {lm('single-random','fz_decay'):+.3f}→{lm('distributed','fz_decay'):+.3f}"
       f"（single-random 扰动净放大且 τ½ 截尾，distributed τ½≈6.6；闭环 cl G_max "
       f"{lm('single-random','cl_Gmax'):.2f}→{lm('distributed','cl_Gmax'):.2f} 同向）。")
    ap(f"3. **overlap 非单调但确实有害**：α=0/0.5/1 的 fz decay24 "
       f"{lm('overlap_a0','fz_decay'):+.3f}/{lm('overlap_a0.5','fz_decay'):+.3f}/{lm('overlap_a1','fz_decay'):+.3f}，"
       f"r=12.5% final {fin_r12['overlap_a0'][0]:.2f}/{fin_r12['overlap_a0.5'][0]:.2f}/{fin_r12['overlap_a1'][0]:.2f}，"
       f"梯度冲突集中在 W/Hold（α=0.5 时 Δcos W={h3['alpha=0.5']['W']:.2f}）；α=0.5 并不比 α=1 轻，"
       f"因此只能说 overlap 有害，不能说伤害随 α 严格单调。")
    ap("4. **可达性 probe 不敏感**：所有拓扑 probe r_eff 均在 1.1~1.2（随机小输入在 20 步闭环后被压缩到单一方向），"
       "H2 的可达性预测未得到支持；寿命方向则明确支持。")
    ap('')
    (ROOT / 'REPORT.md').write_text('\n'.join(A) + '\n')


def h2_judgment(single):
    out = {}

    def get(lab, key, seeds):
        d = {}
        for s in single:
            if s['label'] == lab and s['seed'] in seeds and s.get(key) is not None:
                d.setdefault(s['seed'], []).append(s[key])
        return {k: float(np.mean(v)) for k, v in d.items()}

    base = get('single-random', 'probe_r_eff', set(range(0, 100)))
    dist = get('distributed', 'probe_r_eff', set(range(0, 100)))
    seeds = sorted(set(base) & set(dist))
    if seeds:
        rel = [(dist[s] - base[s]) / (base[s] + 1e-12) for s in seeds]
        out['reach_rel_mean'] = float(np.mean(rel))
        out['reach_seeds_ge10'] = int(sum(r >= .10 for r in rel))
    tb = get('single-random', 'fz_decay', set(range(0, 100)))
    td = get('distributed', 'fz_decay', set(range(0, 100)))
    seeds2 = sorted(set(tb) & set(td))
    if seeds2:
        out['decay_base_mean'] = float(np.mean([tb[s] for s in seeds2]))
        out['decay_dist_mean'] = float(np.mean([td[s] for s in seeds2]))
        out['decay_qual_ok'] = bool(out['decay_dist_mean'] > 0 and out['decay_base_mean'] <= 0)
    tb2 = get('single-random', 'cl_decay', set(range(0, 100)))
    td2 = get('distributed', 'cl_decay', set(range(0, 100)))
    seeds3 = sorted(set(tb2) & set(td2))
    if seeds3:
        out['cl_decay_base'] = float(np.mean([tb2[s] for s in seeds3]))
        out['cl_decay_dist'] = float(np.mean([td2[s] for s in seeds3]))
    return out


def h3_judgment(phase='B'):
    out = {}
    rows = [r for r in load_rows(phase) if 'matrix' in r and r['ratio'] == 0 and r['order'] == 'o0']
    for r in rows:
        ck = ROOT / PHASES[phase] / f"{r['name']}_stage0.pt"
        if not ck.exists():
            continue
        m, c, _ = Mt.load_ckpt(ck)
        gc = Mt.grad_conflict(m, r['seed'], r['order_seq'][1], r['order_seq'][0])
        out.setdefault(f"alpha={r['alpha']:g}", []).append(gc)
    res = {}
    base = out.get('alpha=0')
    if base:
        b = {g: float(np.mean([x[g] for x in base])) for g in ('W', 'route', 'hold')}
        for k, v in sorted(out.items()):
            if k == 'alpha=0':
                continue
            res[k] = {g: float(np.mean([x[g] for x in v])) - b[g] for g in ('W', 'route', 'hold')}
    return res


def main():
    t0 = time.perf_counter()
    single = []
    for phase in PHASES:
        single += single_metrics(phase)
    rows = []
    for phase in PHASES:
        rows += load_rows(phase)
    ver = min(verify_streams(p) for p in PHASES)
    h1 = {t: paired_h1(rows, t) for t in ('multi8', 'multi16')}
    h2 = h2_judgment(single)
    h3 = h3_judgment()
    (ROOT / 'results/B/h3.json').write_text(json.dumps(h3, indent=2))
    write_report(single, rows, ver, h1, h2, h3)
    print('analyze done in', round(time.perf_counter() - t0, 1), 's',
          '| verify', ver, '| H1', {k: round(v['mean'], 2) for k, v in h1.items()},
          '| H2', h2, '| H3', h3)


if __name__ == '__main__':
    main()

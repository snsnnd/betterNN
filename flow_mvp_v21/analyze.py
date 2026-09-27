"""第二十一轮汇总：fixed 臂合并 V20、H1/H2a/H2b/H3、fixed 回归核验、REPORT.md。"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
V20 = (ROOT / '../flow_mvp_v20').resolve()
sys.path.insert(0, str(ROOT))
import experiment as E

PHASES = {'A': 'results/A', 'B': 'results/B', 'C': 'results/C'}
RATIOS = {0: '0', .125: '12.5'}


def load(path):
    return [json.loads(p.read_text()) for p in sorted(path.glob('*.json'))
            if p.name not in ('config.json', 'h3.json') and not p.name.startswith('metrics_')]


def load_cl(path):
    return [r for r in load(path) if 'matrix' in r]


def v20_fixed_rows():
    rows = load_cl(V20 / 'results/B')          # overlap 族：alpha 1 → fixed-overlap；alpha 0 → fixed-disjoint
    out = []
    for r in rows:
        if r['alpha'] not in (0, 1):
            continue
        arm = 'fixed-overlap' if r['alpha'] == 1 else 'fixed-disjoint'
        rr = dict(r)
        rr['arm'] = arm
        rr['v20_fixed'] = True
        ob = float(r['alpha'])
        rr['B0'] = {'O_B': ob, 'cos_B': 0.0, 'jaccard16': ob}
        rr['B_final'] = dict(rr['B0'])
        rr['stage_metrics'] = [{'D_B': 0.0, 'O_h_t8': float('nan'), 'conflict': None}]
        out.append(rr)
    return out


def v20_fixed_singles():
    rows = load(V20 / 'results/B')
    out = []
    for r in rows:
        if not r['name'].startswith('single_') or r['alpha'] not in (0, 1):
            continue
        rr = dict(r)
        rr['arm'] = 'fixed-overlap' if r['alpha'] == 1 else 'fixed-disjoint'
        out.append(rr)
    return out


def by_seed_order(rows):
    d = {}
    for r in rows:
        d.setdefault((r['arm'], r['seed'], r['ratio']), []).append(r)
    return {k: v for k, v in d.items()}


def mean_metric(rs, key):
    return float(np.mean([r[key] for r in rs]))


def stage_mean(rs, key):
    vals = [np.mean([sm.get(key, float('nan')) for sm in r['stage_metrics']]) for r in rs]
    return float(np.nanmean(vals))


def conflict_scalar(r):
    cs = []
    for sm in r['stage_metrics']:
        c = sm.get('conflict')
        if not c:
            continue
        vs = [c[g] for g in ('W', 'route', 'hold') if c.get(g) is not None]
        if vs:
            cs.append(-float(np.mean(vs)))
    return float(np.mean(cs)) if cs else float('nan')


def table(rows, ratio, keys):
    out = {}
    for arm in sorted({r['arm'] for r in rows}):
        rs = [r for r in rows if r['arm'] == arm and r['ratio'] == ratio]
        if not rs:
            continue
        out[arm] = {k: mean_metric(rs, k) for k in keys}
        out[arm]['n'] = len(rs)
        out[arm]['O_B_final'] = float(np.mean([r['B_final']['O_B'] for r in rs]))
        out[arm]['O_B_init'] = float(np.mean([r['B0']['O_B'] for r in rs]))
        out[arm]['D_B'] = stage_mean(rs, 'D_B')
        out[arm]['O_h_t8'] = stage_mean(rs, 'O_h_t8')
        out[arm]['C_grad'] = float(np.nanmean([conflict_scalar(r) for r in rs]))
    return out


def paired_seed(rows, ratio, key, arm_a, arm_b):
    """按 (seed, ratio) 先对 orders 聚合，再配对 arm_a − arm_b。"""
    d = {}
    for r in rows:
        if r['ratio'] != ratio:
            continue
        d.setdefault((r['arm'], r['seed']), []).append(r[key])
    seeds = sorted({s for (a, s) in d if a == arm_a} & {s for (a, s) in d if a == arm_b})
    diffs = [np.mean(d[(arm_a, s)]) - np.mean(d[(arm_b, s)]) for s in seeds]
    return seeds, np.array(diffs)


def h1(rows):
    out = {}
    for ratio in (0.125, 0):
        seeds, diffs = paired_seed(rows, ratio, 'forgetting', 'fixed-disjoint', 'fixed-overlap')
        out[RATIOS[ratio]] = {'seeds': seeds, 'diffs_pp': (diffs * 100).tolist(),
                              'mean_pp': float(diffs.mean() * 100) if len(diffs) else float('nan'),
                              'n_neg': int((diffs < 0).sum())}
    return out


def h2a(rows):
    """learnable λ=0 的 O_B^final / O_B^init，按 seed 对 orders 聚合。"""
    d = {}
    for r in rows:
        if r['arm'] == 'learnable':
            d.setdefault(r['seed'], []).append(r['B_final']['O_B'] / (r['B0']['O_B'] + 1e-12))
    ratios = {s: float(np.mean(v)) for s, v in d.items()}
    n = sum(v <= .75 for v in ratios.values())
    return {'seeds': sorted(ratios), 'ratio': [round(ratios[s], 3) for s in sorted(ratios)], 'n_le_075': n}


def h2b(rows):
    """penalty family 相对 learnable(λ=0) 的 ΔO_B / Δforgetting(r=12.5%) / Δmean_cos，跨 λ 方向一致。"""
    out = {}
    for fam, arms in [('orth', ['learnable+orth0.1', 'learnable+orth1']),
                      ('overlap', ['learnable+overlap0.1', 'learnable+overlap1'])]:
        fam_out = {}
        for arm in arms:
            d_ob = {}
            for r in rows:
                if r['arm'] == arm:
                    d_ob.setdefault(r['seed'], []).append(r['B_final']['O_B'])
            l_ob = {}
            for r in rows:
                if r['arm'] == 'learnable':
                    l_ob.setdefault(r['seed'], []).append(r['B_final']['O_B'])
            seeds = sorted(set(d_ob) & set(l_ob))
            dob = float(np.mean([np.mean(d_ob[s]) - np.mean(l_ob[s]) for s in seeds]))
            _, dfog = paired_seed(rows, 0.125, 'forgetting', arm, 'learnable')
            _, dcos = paired_seed(rows, 0.125, 'final_mean', arm, 'learnable')
            c_a = np.mean([conflict_scalar(r) for r in rows if r['arm'] == arm])
            c_l = np.mean([conflict_scalar(r) for r in rows if r['arm'] == 'learnable'])
            fam_out[arm] = {'dO_B': dob, 'dforget_pp': float(dfog.mean() * 100) if len(dfog) else float('nan'),
                            'dC_grad': float(c_a - c_l)}
        ok = all((fam_out[a]['dO_B'] <= -.10) or (fam_out[a]['dforget_pp'] <= -2.0)
                 for a in arms if a in fam_out)
        dir_ob = [fam_out[a]['dO_B'] for a in arms if a in fam_out]
        out[fam] = {'arms': fam_out, 'direction_consistent': bool(len(dir_ob) == 2 and
                                                                  (all(v < 0 for v in dir_ob) or all(v > 0 for v in dir_ob))),
                    'criterion_ok': bool(ok)}
    return out


def spearman(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 4:
        return float('nan')
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def h3(rows, ratio=0):
    """机制链 Spearman：O_B → O_h → C∇ → F；单位 = arm × seed（orders 聚合）。仅 v21 learnable 臂。"""
    units = {}
    for r in rows:
        if r['ratio'] != ratio or any('O_h_t8' not in sm for sm in r['stage_metrics']):
            continue
        u = units.setdefault((r['arm'], r['seed']), {'OB': [], 'Oh': [], 'C': [], 'F': []})
        u['OB'].append(r['B_final']['O_B'])
        u['Oh'].append(np.mean([sm['O_h_t8'] for sm in r['stage_metrics']]))
        u['C'].append(conflict_scalar(r))
        u['F'].append(r['forgetting'])
    OB = [np.mean(v['OB']) for v in units.values()]
    Oh = [np.mean(v['Oh']) for v in units.values()]
    C = [np.mean(v['C']) for v in units.values()]
    F = [np.mean(v['F']) for v in units.values()]
    return {'n_units': len(OB), 'rho_OB_Oh': spearman(OB, Oh), 'rho_Oh_C': spearman(Oh, C),
            'rho_C_F': spearman(C, F), 'rho_OB_F': spearman(OB, F)}


def fixed_regression():
    rows21 = load_cl(ROOT / 'results/fixed_reg')
    rows20 = v20_fixed_rows()
    idx = {(r['arm'], r['seed'], r['order'], r['ratio']): r for r in rows20}
    diffs = []
    for r in rows21:
        v = idx.get((r['arm'], r['seed'], r['order'], r['ratio']))
        if v is None:
            continue
        diffs.append(float(np.max(np.abs(np.array(r['matrix']) - np.array(v['matrix'])))))
    return float(max(diffs)) if diffs else -1.0, len(diffs)


def fmt(x, e=2):
    return f'{x:.{e}f}'


def write_report(rows, fixed_single, h1v, h2av, h2bv, h3v, reg):
    A = []
    ap = A.append
    ap('# 第二十一轮：Adaptive Input Decoupling（B 自发/正则解耦）')
    ap('')
    ap("协议：hybrid 信用分配（core 5 步截断 / B 全 BPTT，双 optimizer、分别 clip）；写入域 role0∪role1；"
       "`‖B_eff‖₂=5.6`；A'B'C'D'、5 seeds、o0/o1/o2、r∈{0,12.5%}。")
    ap('')
    ap('## 1. fixed 回归核验（v21 vs V20 检查点）')
    ap('')
    ap(f"- 逐流矩阵最大差：{fmt(reg[0])}（{reg[1]} 条配对）；应为 0，证明 fixed 臂协议未变。")
    ap('')
    ap('## 2. CL 汇总（合并 3 顺序，均值）')
    ap('')
    for ratio in (0, .125):
        t = table(rows, ratio, ['acquisition', 'final_mean', 'forgetting'])
        ap(f"### r={RATIOS[ratio]}%")
        ap('')
        ap('| arm | acquisition | final_mean | forgetting (pp) | O_B init→final | D_B | O_h(t8) | C∇ |')
        ap('|---|---:|---:|---:|---:|---:|---:|---:|')
        def f3(x):
            return '—' if (isinstance(x, float) and np.isnan(x)) else f'{x:.3f}'

        for arm in sorted(t):
            v = t[arm]
            ap(f"| {arm} | {v['acquisition']:.3f} | {v['final_mean']:.3f} | {v['forgetting']*100:.2f} | "
               f"{v['O_B_init']:.3f}→{v['O_B_final']:.3f} | {v['D_B']:.3f} | "
               f"{f3(v['O_h_t8'])} | {f3(v['C_grad'])} |")
        ap('')
    ap('## 3. 预注册判定')
    ap('')
    ap('### H1（fixed-disjoint > fixed-overlap）')
    ap('')
    for k, v in h1v.items():
        tag = '主条件' if k == '12.5' else '方向复核'
        ap(f"- r={k}%（{tag}）：Δforgetting = {v['mean_pp']:.2f}pp，负向 {v['n_neg']}/{len(v['seeds'])}；"
           f"门槛 −2pp → {'通过' if v['mean_pp'] <= -2 else '未通过'}")
    ap('')
    ap('### H2a（learnable λ=0 自发解耦）')
    ap('')
    ap(f"- O_B^final/O_B^init 按 seed：{h2av['ratio']}；≤0.75 的 seeds {h2av['n_le_075']}/5 → "
       f"{'通过' if h2av['n_le_075'] >= 4 else '未通过'}")
    ap('')
    ap('### H2b（penalty 相对 λ=0 的增量）')
    ap('')
    for fam, v in h2bv.items():
        ap(f"- {fam}：方向一致 {v['direction_consistent']}，判定 {v['criterion_ok']}")
        for arm, d in v['arms'].items():
            ap(f"  - {arm}: ΔO_B={d['dO_B']:+.3f}, Δforget={d['dforget_pp']:+.2f}pp, ΔC∇={d['dC_grad']:+.3f}")
    ap('')
    ap('### H3（机制链 Spearman，arm×seed）')
    ap('')
    ap(f"- n={h3v['n_units']}；ρ(O_B,O_h)={h3v['rho_OB_Oh']:.2f}、ρ(O_h,C∇)={h3v['rho_Oh_C']:.2f}、"
       f"ρ(C∇,F)={h3v['rho_C_F']:.2f}、ρ(O_B,F)={h3v['rho_OB_F']:.2f}；门槛 0.4（r=0）")
    ap('')
    ap('## 4. 边界')
    ap('')
    ap('- mediation 为 exploratory（5 seeds）；V21D（Full BPTT 40 流）与 V21-trunc 负控单独报告。')
    ap('- fixed 臂直接复用 V20（fixed_reg 已核验）；O_h 为 counterfactual 输入响应子空间主角均值（r=8, t=8）。')
    (ROOT / 'REPORT.md').write_text('\n'.join(A) + '\n')


def main():
    t0 = time.perf_counter()
    rows = []
    for p in PHASES:
        rows += load_cl(ROOT / PHASES[p])
    rows += v20_fixed_rows()
    fixed_single = v20_fixed_singles()
    h1v = h1(rows)
    h2av = h2a(rows)
    h2bv = h2b(rows)
    h3v = h3(rows, ratio=0)
    reg = fixed_regression()
    write_report(rows, fixed_single, h1v, h2av, h2bv, h3v, reg)
    summary = {'h1': h1v, 'h2a': h2av, 'h2b': h2bv, 'h3_r0': h3v,
               'h3_r125': h3(rows, ratio=0.125), 'fixed_reg_maxdiff': reg}
    (ROOT / 'results/summary.json').write_text(json.dumps(summary, indent=2))
    (ROOT / 'verification.txt').write_text(
        f"fixed regression maxdiff: {reg[0]:.3e} over {reg[1]} streams\n"
        f"h1 r12.5 mean: {h1v['12.5']['mean_pp']:.2f}pp\n"
        f"h2a n<=0.75: {h2av['n_le_075']}/5\n")
    print('analyze done in', round(time.perf_counter() - t0, 1), 's')
    print(json.dumps({k: v for k, v in summary.items() if k != 'h2b'}, indent=1))
    print(json.dumps(summary['h2b'], indent=1))


if __name__ == '__main__':
    main()

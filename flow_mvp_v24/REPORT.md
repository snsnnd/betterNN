# 第二十四轮：Input Write Dynamics（等能量写入时间结构）

协议：z_t=Σa_k B x_{t-k}（因果 FIR，Σa²=1；single=[1]、burst3/5、decay-fast/slow）；Phase A 用 V20 fixed-disjoint 固定 B、延迟任务 T{20,40,80,160}、65 epochs、5 seeds × 4 tasks；Phase B 用 V23 Chain-select T=80（learnable B hybrid / core K=5）、500 epochs。

## 0. 结论摘要

1. **等能量下的写入时间结构确实有作用，但只在长 T 才显著（延迟任务）**：T=20/40 无差异；T=80 最优 kernel 仅 +3.5pp（burst5，17/20 同向）；T=160 +5.3pp（burst5，16/20）。严格预注册的 H1@T=80 未通过。
2. **优势随 T 单调增长**（H2 通过）：gap = −0.0 / +1.1 / +3.5 / +5.3pp（T=20/40/80/160），不符合“窗口对齐假象”。
3. **在 V23 Chain-select 上写入结构是关键（H3 通过）**：T=80、core 仍为 K=5、B 仍为 hybrid 全信用时，`single` 0.777（xor）/0.913（xorsw）→ `burst5` **0.971** / `decay-slow` **0.999**；且把 single 的崩溃 seed 全部救回（xor 5/5 可达、xorsw 5/5）。**V23 的 soft-selection 失败很大程度源于单次瞬时写入。**
4. **偏好 kernel 依赖任务**（xor→burst5、xorsw→decay-slow），说明这是“写入时间策略”的自由度，支持下一步 V24B 的可学习 write gate / scheduler（而不是固定 kernel）。
5. **机制不是幅度也不是中程可解码性**：||Δh|| retention 与 acc 不同向（全网格 r=0.08；T≥80 子集 r=0.42）；probe 在 T/2 与 acc 负相关（r=−0.47）、在 T−1 弱正相关（r=+0.46）——kernel 改善的是“末端状态几何/可读出性”，不是简单地把扰动留得更久。

- Phase A T=20: single=0.994，最优 kernel decay-slow=0.994（Δ=-0.0pp）
- Phase A T=40: single=0.976，最优 kernel decay-slow=0.987（Δ=+1.1pp）
- Phase A T=80: single=0.894，最优 kernel burst5=0.929（Δ=+3.5pp）
- Phase A T=160: single=0.673，最优 kernel burst5=0.726（Δ=+5.3pp）
- Phase 0 回归：single@T20 vs V20 fixed-disjoint singles max|Δfinal| = 0.000e+00（20 条配对，应为 0）

## 1. Phase A：acc 汇总（4 tasks × 5 seeds）

| T | single | burst3 | burst5 | decay-fast | decay-slow |
|---|---:|---:|---:|---:|---:|
| 20 | 0.994±0.001 | 0.992±0.002 | 0.993±0.001 | 0.993±0.001 | 0.994±0.001 |
| 40 | 0.976±0.007 | 0.986±0.004 | 0.977±0.014 | 0.982±0.006 | 0.987±0.003 |
| 80 | 0.894±0.022 | 0.880±0.025 | 0.929±0.019 | 0.886±0.024 | 0.926±0.018 |
| 160 | 0.673±0.018 | 0.700±0.017 | 0.726±0.021 | 0.699±0.018 | 0.724±0.021 |

| T | kernel | Δacc vs single (pp) | 同向 | ret_half 几何均值比 |
|---|---:|---:|---:|---:|
| 20 | burst3 | -0.21 | 7/20 | -27.6% |
| 20 | burst5 | -0.08 | 8/20 | -39.2% |
| 20 | decay-fast | -0.04 | 10/20 | -28.1% |
| 20 | decay-slow | -0.02 | 9/20 | -38.4% |
| 40 | burst3 | +1.02 | 14/20 | -20.6% |
| 40 | burst5 | +0.13 | 13/20 | -43.8% |
| 40 | decay-fast | +0.63 | 12/20 | -29.1% |
| 40 | decay-slow | +1.05 | 11/20 | -38.4% |
| 80 | burst3 | -1.40 | 14/20 | -36.2% |
| 80 | burst5 | +3.47 | 17/20 | -34.9% |
| 80 | decay-fast | -0.83 | 14/20 | -39.7% |
| 80 | decay-slow | +3.21 | 17/20 | -35.5% |
| 160 | burst3 | +2.66 | 15/20 | -43.4% |
| 160 | burst5 | +5.32 | 16/20 | -28.7% |
| 160 | decay-fast | +2.60 | 16/20 | -46.1% |
| 160 | decay-slow | +5.10 | 16/20 | -31.8% |

## 2. Retention（t=0.5T / T−1，相对 t=5；衰减斜率）

| T | kernel | ret_half | ret_end | decay_slope | lifetime |
|---|---:|---:|---:|---:|---:|
| 20 | single | 4.405 | 12.986 | +0.1770 | nan |
| 20 | burst3 | 3.165 | 8.404 | +0.1501 | nan |
| 20 | burst5 | 2.659 | 6.810 | +0.1387 | nan |
| 20 | decay-fast | 3.180 | 9.680 | +0.1664 | nan |
| 20 | decay-slow | 2.700 | 6.959 | +0.1402 | nan |
| 40 | single | 7.797 | 23.534 | +0.0931 | nan |
| 40 | burst3 | 5.567 | 12.485 | +0.0748 | nan |
| 40 | burst5 | 4.034 | 8.614 | +0.0636 | nan |
| 40 | decay-fast | 5.158 | 12.375 | +0.0775 | nan |
| 40 | decay-slow | 4.428 | 9.310 | +0.0662 | nan |
| 80 | single | 5.756 | 19.563 | +0.0402 | nan |
| 80 | burst3 | 3.971 | 9.464 | +0.0300 | nan |
| 80 | burst5 | 3.701 | 7.400 | +0.0287 | nan |
| 80 | decay-fast | 3.529 | 8.692 | +0.0305 | nan |
| 80 | decay-slow | 3.734 | 7.871 | +0.0302 | nan |
| 160 | single | 2.434 | 9.228 | +0.0206 | nan |
| 160 | burst3 | 0.602 | 3.824 | +0.0135 | nan |
| 160 | burst5 | 1.162 | 3.780 | +0.0118 | nan |
| 160 | decay-fast | 0.605 | 3.786 | +0.0135 | nan |
| 160 | decay-slow | 1.101 | 3.902 | +0.0119 | nan |

## 2.5 状态可解码性 probe（线性 probe，前 512 拟合 / 后 512 评估，两输出平均）

| T | kernel | t=5 | t=T/2 | t=T−1 |
|---|---:|---:|---:|---:|
| 80 | single | 0.837 | 0.826 | 0.860 |
| 80 | burst3 | 0.838 | 0.861 | 0.884 |
| 80 | burst5 | 0.813 | 0.867 | 0.894 |
| 80 | decay-fast | 0.763 | 0.841 | 0.869 |
| 80 | decay-slow | 0.787 | 0.864 | 0.884 |
| 160 | single | 0.722 | 0.730 | 0.675 |
| 160 | burst3 | 0.708 | 0.705 | 0.676 |
| 160 | burst5 | 0.713 | 0.704 | 0.712 |
| 160 | decay-fast | 0.726 | 0.707 | 0.675 |
| 160 | decay-slow | 0.721 | 0.712 | 0.713 |

## 3. Phase B：Chain-select T=80（K=5, learnable B hybrid）

| task | kernel | final | best-val | reach(≥0.9) |
|---|---:|---:|---:|---:|
| xor | single | 0.777 | 0.970 | 4/5 |
| xor | burst5 | 0.971 | 0.998 | 5/5 |
| xor | decay-slow | 0.912 | 0.966 | 4/5 |
| xorsw | single | 0.913 | 0.997 | 5/5 |
| xorsw | burst5 | 0.913 | 1.000 | 5/5 |
| xorsw | decay-slow | 0.999 | 1.000 | 5/5 |

## 4. 预注册判定

- **H1**：写入时间结构有效（T=80：Δacc≥+5pp 或 retention≥+20%）：T=80: best=burst5 Δacc=+3.5pp Δret_half=-34.9%；T=160: best=burst5 Δacc=+5.3pp Δret_half=-28.7% → 未通过
- **H1null**：H1-null（所有非 single：|Δacc|≤2pp 且 |Δret|≤10%） → 未通过
- **H2**：gap(T160)≥gap(T20)（T20=-0.0pp、T40=+1.1pp、T80=+3.5pp、T160=+5.3pp） → **通过**
- **H4**：||dh|| ret_half 与 acc 同向：全网格 n=16 r=0.08（T≥80 子集 n=8 r=0.42）；probe 相关：T/2 r=-0.47、T−1 r=0.46 → 未通过
- **H3**：Chain-select T=80 最优 kernel ≥ single+5pp 且 ≥0.90：xor: best=burst5 Δ=+19.4pp acc=0.971；xorsw: best=decay-slow Δ=+8.6pp acc=0.999 → **通过**

## 5. 边界

- Phase A 固定 B（V20 fixed-disjoint）与核心训练协议；kernel 固定不训练，验证时间结构自由度本身。
- Σa²=1 等总能量；无幅度扫描。
- Phase B 的 single 是 V23 同 cell 回归锚点；结论只针对该任务族。

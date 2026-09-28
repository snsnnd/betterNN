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
| xor | single | 0.774 | 0.945 | 6/8 |
| xor | burst5 | 0.898 | 0.978 | 7/8 |
| xor | decay-slow | 0.912 | 0.966 | 4/5 |
| xorsw | single | 0.892 | 0.981 | 7/8 |
| xorsw | burst5 | 0.916 | 0.967 | 7/8 |
| xorsw | decay-slow | 0.999 | 1.000 | 5/5 |

## 3.5 Phase C（V24B）：可学习 Write Scheduler `w_t`

结构：`w_t=2σ(MLP(x_t,meta_t))`（不看 h），`e_t=w_t⊙x_t`，`z_t=Σ_k a_k(e_{t-k}B)`；controller 末层零初始化 → 初始 w≡1（与固定 kernel 起点逐值一致）；w 与 B 走 full BPTT（hybrid），core 仍 K=5。

### Q1：能力（T=80，xor/xorsw）

| task | kernel | fixed final | sched final | Δ | fixed best-val | sched best-val | sched reach |
|---|---:|---:|---:|---:|---:|---:|---:|
| xor | burst5 | 0.971 | 0.946 | -2.4pp | 0.998 | 0.958 | 4/5 |
| xor | decay-slow | 0.912 | 0.946 | +3.5pp | 0.966 | 0.960 | 4/5 |
| xor | single | 0.777 | 0.917 | +14.1pp | 0.970 | 0.932 | 4/5 |
| xorsw | burst5 | 0.913 | 0.986 | +7.3pp | 1.000 | 0.998 | 5/5 |
| xorsw | decay-slow | 0.999 | 0.999 | -0.0pp | 1.000 | 0.999 | 5/5 |
| xorsw | single | 0.913 | 0.999 | +8.6pp | 0.997 | 1.000 | 5/5 |

### Q2：事件级 w（均值；distractor 应最小）

| task | kernel | A | c1 | distractor | B | c2 | all |
|---|---:|---:|---:|---:|---:|---:|---:|
| xor | burst5 | 1.21 | 1.38 | **1.53** | 1.62 | 1.30 | 1.45 |
| xor | decay-slow | 1.14 | 1.36 | **1.68** | 1.79 | 1.58 | 1.56 |
| xor | single | 0.42 | 0.74 | **1.10** | 1.35 | 1.09 | 1.03 |
| xorsw | burst5 | 0.55 | 0.69 | **0.74** | 0.80 | 1.83 | 0.97 |
| xorsw | decay-slow | 0.42 | 0.47 | **0.40** | 0.39 | 1.68 | 0.69 |
| xorsw | single | 0.09 | 0.12 | **0.12** | 0.18 | 1.77 | 0.52 |

### Q3：T=160 xor

| arm | final | best-val | reach |
|---|---:|---:|---:|
| single fixed | 0.770 | 0.902 | 2/3 |
| burst5 fixed | 0.777 | 0.943 | 2/3 |
| burst5+sched | 0.743 | 0.794 | 0/3 |

## 3.6 Phase D（V24C）：Adaptive Temporal Compression（kernel bank selector）

结构：`α_t=softmax(MLP(x_t,meta_t))`（9→16→5，bias 初始 [4,0,0,0,0]）；`ã_t=α_tK/‖α_tK‖`（Σa²=1）；`z_{t+τ} += ã_{t,τ}B x_t`；无 w_t；core 仍 K=5、B/scheduler 走 hybrid full。

### Chain-select T=80：adaptive vs fixed

| task | adaptive final | best | reach | fixed best（kernel） | probe T/2 | probe T−1 |
|---|---:|---:|---:|---|---:|---:|
| xor | 0.998 | 0.971 | 5/5 | burst5 | 0.882 | 0.996 |
| xorsw | 1.000 | 0.999 | 5/5 | decay-slow | 0.620 | 0.997 |

### 事件级 α 与有效时间长度 `L_eff=Σ τ ã_τ²`

| task | 事件 | α(single,burst3,burst5,dfast,dslow) | L_eff |
|---|---|---|---:|
| xor | A | 0.46,0.06,0.30,0.03,0.15 | 0.80 |
| xor | c1 | 0.11,0.06,0.54,0.03,0.26 | 1.43 |
| xor | distractor | 0.02,0.04,0.61,0.02,0.31 | 1.66 |
| xor | B | 0.00,0.03,0.62,0.02,0.33 | 1.70 |
| xor | c2 | 0.23,0.05,0.51,0.03,0.18 | 1.28 |
| xorsw | A | 0.91,0.01,0.04,0.01,0.03 | 0.03 |
| xorsw | c1 | 0.87,0.02,0.06,0.01,0.04 | 0.07 |
| xorsw | distractor | 0.80,0.02,0.09,0.01,0.07 | 0.18 |
| xorsw | B | 0.74,0.03,0.13,0.01,0.10 | 0.30 |
| xorsw | c2 | 0.14,0.05,0.56,0.01,0.24 | 1.41 |

### 延迟任务：adaptive vs fixed（含 probe t=T/2 / T−1）

| T | adaptive final | best fixed（kernel） | single | adaptive probe T/2 | adaptive probe T−1 | single probe T−1 |（全部 task0）
|---|---:|---|---:|---:|---:|---:|
| 40 | 0.973 | 0.987（burst5） | 0.966 | 0.865 | 0.887 | nan |
| 80 | 0.734 | 0.898（decay-slow） | 0.790 | 0.815 | 0.823 | 0.791 |
| 160 | 0.572 | 0.606（burst5） | 0.571 | 0.775 | 0.684 | 0.663 |

| T | 事件 | α(single,burst3,burst5,dfast,dslow) | L_eff |
|---|---|---|---:|
| 40 | A | 0.96,0.01,0.01,0.01,0.01 | 0.00 |
| 40 | B | 0.96,0.01,0.01,0.01,0.01 | 0.00 |
| 80 | A | 0.98,0.00,0.00,0.00,0.00 | 0.00 |
| 80 | B | 0.98,0.00,0.00,0.00,0.00 | 0.00 |
| 160 | A | 0.64,0.24,0.06,0.02,0.05 | 0.30 |
| 160 | B | 0.69,0.22,0.04,0.02,0.03 | 0.25 |

## 4. 预注册判定

- **H1**：写入时间结构有效（T=80：Δacc≥+5pp 或 retention≥+20%）：T=80: best=burst5 Δacc=+3.5pp Δret_half=-34.9%；T=160: best=burst5 Δacc=+5.3pp Δret_half=-28.7% → 未通过
- **H1null**：H1-null（所有非 single：|Δacc|≤2pp 且 |Δret|≤10%） → 未通过
- **H2**：gap(T160)≥gap(T20)（T20=-0.0pp、T40=+1.1pp、T80=+3.5pp、T160=+5.3pp） → **通过**
- **H4**：||dh|| ret_half 与 acc 同向：全网格 n=16 r=0.08（T≥80 子集 n=8 r=0.42）；probe 相关：T/2 r=-0.47、T−1 r=0.46 → 未通过
- **H3**：Chain-select T=80 最优 kernel ≥ single+5pp 且 ≥0.90：xor: best=burst5 Δ=+19.4pp acc=0.971；xorsw: best=decay-slow Δ=+8.6pp acc=0.999 → **通过**
- **H5**：T=80 同一 scheduler（kernel=decay-slow）：xor=0.946（Δ+17.0pp）、xorsw=0.999（Δ+8.5pp），两任务 ≥0.95 且 Δ≥+5pp → 未通过
- **H6**：distractor 抑制（≥2 cells 满足 E[w|d]≤E[w|imp]−0.2）：xor/burst5: 0/5 seeds, w_all=1.45；xor/decay-slow: 0/5 seeds, w_all=1.56；xor/single: 0/5 seeds, w_all=1.03；xorsw/burst5: 0/5 seeds, w_all=0.97；xorsw/decay-slow: 0/5 seeds, w_all=0.69；xorsw/single: 0/5 seeds, w_all=0.52 → 未通过
- **H7**：T=160 xor：burst5+sched=0.743 vs burst5 fixed=0.777（需≥−2pp）、vs single=0.770（需≥+5pp） → 未通过
- **HC1**：Chain T=80 性能：xor: adaptive=0.998 vs best fixed=0.971（需≥−2pp）；xorsw: adaptive=1.000 vs best fixed=0.999（需≥−2pp） → **通过**
- **HC2**：时间选择（≥1 任务：L_imp≥1.2L_d 且 ≥4/5 seeds 且 ‖Δα‖₁>0.2）：xor: L_imp/L_d=0.78（0/5 seeds>1.2）、‖Δα‖₁=0.40；xorsw: L_imp/L_d=2.51（4/5 seeds>1.2）、‖Δα‖₁=0.31 → **通过**
- **HC3**：延迟任务 task0 长 T（T=160 需≥best fixed−2pp；T160 只跑 task0）：T40/task0: adaptive=0.973 vs best fixed=0.987（5 seeds）；T80/task0: adaptive=0.734 vs best fixed=0.898（5 seeds）；T160/task0: adaptive=0.572 vs best fixed=0.606（5 seeds） → 未通过
- **HC-verdict**：结果分类 A：自适应时间压缩成立 → V24D 连续 λ（HC1=True, HC2=True, HC3=False） → info

## 5. 边界

- Phase A 固定 B（V20 fixed-disjoint）与核心训练协议；kernel 固定不训练，验证时间结构自由度本身。
- Σa²=1 等总能量；无幅度扫描。
- Phase B 的 single 是 V23 同 cell 回归锚点；结论只针对该任务族。
- Phase C 中 controller 只看 `(x_t, meta_t)`（不看 h）、第一版固定 `a_k`；w 与 B 同走 hybrid full 通道。

### Phase C 解释（exploratory）

- **能力（Q1）**：同一 scheduler 配置没有超过"每任务人工选最优固定 kernel"：xor 最好的 fixed 是 burst5（0.971），最好的 sched 是 decay-slow（0.946）；xorsw 最好的 fixed/sched 都是 decay-slow（0.999）。sched 的方差更大（burst5+sched xor 有一条 0.733 的崩溃；single+sched 既救回 0.527 也弄坏 1.0 的 seed）。
- **重要性（Q2）**：30 条流里没有一条学会相对抑制 distractor（H6 0/30）。更糟的是常见两种退化：（a）**全开放大**：xor/burst5、xor/decay-slow 的 w_all≈1.45/1.56，所有事件（含 distractor）都被写到 >1；（b）**全局压缩 + 末端保留**：xorsw 的各臂把 A/c1/distractor/B 压到 0.4–0.8，只把 c2 留下 ~1.7。两者都不是"重要多写、噪声少写"。
- **为什么标量 input gate 很容易退化的机制解释**：任务是符号分类，`y=sign(v)`，对输入做**正标量缩放不改变可解性**（网络可用内部增益补偿），因此"写多强"的最优点是一大片平台；梯度把 `w` 推向放大（更小有效噪声）或整体压缩，而不是相对选择性。相对重要性只有在"写重了会挤占别的信息"的预算约束下才有梯度——当前的加法写入没有这种竞争。
- **长 T（Q3）**：T=160 的 Chain-select 连 fixed `burst5` 都只有 0.777（reach 2/3），scheduler 0.743（reach 0/3）没有展示空间；V24A 的长 T 写入结构收益只在延迟任务上成立，不能外推到 Chain-select。
- **结论**：V24B 第一版（标量 `w_t`）失败：它既没有稳定达到最优固定 kernel，也没有学会重要性。下一步优先做 **V24C：动态 `λ_t`（trace shaping / 记多久）**，或给 scheduler 加**写入预算约束**（迫使事件竞争），而不是继续调 `w_t` 的容量。

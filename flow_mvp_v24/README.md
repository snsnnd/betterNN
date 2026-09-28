# 第二十四轮：Input Write Dynamics（等能量写入时间结构）

- 设计见 [PLAN.md](PLAN.md)（含运行期诊断 Addendum）；结果见 [REPORT.md](REPORT.md)（`analyze.py` 自动生成）。
- 问题：当前是“单次瞬时写入”（信息只在 t=0/1 通过 `B x_t` 一次倒进状态）。**同样信息、同样总 L2 写入能量**，把写入改成有时间结构的内部 trace `z_t=Σ_k a_k B x_{t-k}`（因果 FIR，`Σa²=1`，外部输入仍只出现一次），会不会更稳？
- Phase A：固定 B（V20 `fixed-disjoint`）、延迟任务 A'B'C'D'、T∈{20,40,80,160}、5 kernels × 4 tasks × 5 seeds；Phase B：V23 Chain-select T=80、保持 learnable B hybrid / core K=5，只换 kernel。

## 主要结果

- **Phase A（延迟任务）**：`single` 与 V20 fixed-disjoint 逐位一致（20/20，max|Δ|=0）。写入结构在 T≤40 无差异；T=80 最优 `burst5` +3.5pp（17/20 同向）；**T=160 +5.3pp（burst5，16/20）**。严格预注册 H1@T=80 未通过。
- **H2 通过：优势随 T 单调增长**（−0.0 / +1.1 / +3.5 / +5.3pp @ T=20/40/80/160）。
- **H3 通过（更重要）：Chain-select T=80、core 仍 K=5、B 仍 hybrid 全信用时**：
  - xor：`single` 0.777 → **`burst5` 0.971（+19.4pp）**，5/5 可达；
  - xorsw：`single` 0.913 → **`decay-slow` 0.999（+8.6pp）**，5/5 可达；
  - `single` 与 V23 同 cell 逐位一致（10/10）。**V23 的 soft-selection 崩溃很大程度来自单次瞬时写入，而不是 core 信用窗口。**
- **机制不是“把扰动留得更久”**：||Δh|| retention 与 acc 不同向（全网格 r=0.08；T≥80 子集 r=0.42）；probe 可解码性在 T/2 负相关（r=−0.47）、T−1 弱正相关（r=+0.46）——kernel 改善的是末端状态几何/可读出性。
- **偏好 kernel 依赖任务**（xor→burst5、xorsw→decay-slow）→ 写入时间策略需要“学习”而不是人工固定；Phase C 试了标量 write gate，但得到明确的负结果（见下）。

## Phase C（V24B）：可学习 Write Scheduler `w_t` —— 负结果

结构：`w_t=2σ(MLP(x_t,meta_t))`（不看 h），`e_t=w_t⊙x_t`，`z_t=Σa_k(e_{t-k}B)`；controller 末层零初始化（初始 w≡1，与固定 kernel 起点逐值一致）；w 与 B 走 full BPTT（hybrid），core 仍 K=5。

- **Q1 能力：未过**（H5）。同一 scheduler 配置没有超过“每任务人工选最优固定 kernel”：xor 最好 fixed `burst5` 0.971 vs 最好 sched `decay-slow` 0.946；xorsw 两者都是 `decay-slow` 0.999。sched 方差更大（一条 0.733 崩溃；single+sched 既救回 0.527 也弄坏 1.0 的 seed）。
- **Q2 重要性：未过**（H6，0/30 runs）。没有一条流相对抑制 distractor；常见退化是（a）**全开放大**（xor 臂 w_all≈1.45–1.56，含 distractor 全部 >1）；（b）**只留末端事件**（xorsw 臂把 A/c1/distractor/B 压到 0.4–0.8，只留 c2≈1.7）。T=160 更明显：不同 seed 分别杀掉 A、c2、B。
- **机制解释**：任务是符号分类，`y=sign(v)` 对输入做**正标量缩放不改变可解性**（内部增益可补偿），所以“写多强”的最优点是一大片平台；加法写入没有“写重了会挤占别的信息”的竞争，相对重要性没有梯度。要让它有意义，需要**写入预算/竞争约束**或改做 **`λ_t`（trace shaping）**。
- **Q3 长 T：未过**（H7）。T=160 xor：burst5+sched final 0.743（best-val 0/3 可达）≈ fixed single 0.770，不能恢复长 T 行为。

结论：标量 input gate 是错误的第一自由度；V24C 应转向动态 `λ_t`/带预算的 scheduler，而不是继续调 `w_t`。

## 运行

```bash
# Phase A：延迟任务 × 固定 B × 5 kernels
FLOW_THREADS=1 python experiment.py --phase delay --kernels single burst3 burst5 decay-fast decay-slow \
    --Ts 20 40 80 160 --tasks 0 1 2 3 --seeds 11 22 33 44 55 --epochs 65 --jobs 8 --out results/delay
# Phase B：Chain-select T=80 × K=5 hybrid
FLOW_THREADS=1 python experiment.py --phase chain --kernels single burst5 decay-slow \
    --Ts 80 --seeds 11 22 33 44 55 --epochs 500 --jobs 6 --out results/chain
# Phase C（V24B）：可学习 write scheduler
FLOW_THREADS=1 python scheduler.py --kernels single burst5 decay-slow --tasks xor xorsw \
    --Ts 80 --seeds 11 22 33 44 55 --epochs 500 --jobs 6 --out results/sched_chain
FLOW_THREADS=1 python scheduler.py --kernels burst5 --tasks xor \
    --Ts 160 --seeds 11 22 33 --epochs 500 --jobs 1 --out results/sched_chain
python analyze.py && python plot.py     # REPORT.md / write_dynamics.png / chain_kernels.png / scheduler.png
python verify_sched.py                  # 初始化等价 + w≡1 + hybrid 梯度检查
```

kernel 定义：`single=[1]`、`burst3/5` 均匀、`decay-fast/slow ∝ 0.5^k / 0.85^k`（L2 归一）。Phase A 固定 B、核心按 V20 协议训练；Phase B 用 hybrid（B full / core K=5），`single` 为 V23 回归锚点。

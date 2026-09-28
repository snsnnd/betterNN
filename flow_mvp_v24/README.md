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
- **偏好 kernel 依赖任务**（xor→burst5、xorsw→decay-slow）→ 下一步应做**可学习 write gate / scheduler（V24B）**，而不是固定 kernel。

## 运行

```bash
# Phase A：延迟任务 × 固定 B × 5 kernels
FLOW_THREADS=1 python experiment.py --phase delay --kernels single burst3 burst5 decay-fast decay-slow \
    --Ts 20 40 80 160 --tasks 0 1 2 3 --seeds 11 22 33 44 55 --epochs 65 --jobs 8 --out results/delay
# Phase B：Chain-select T=80 × K=5 hybrid
FLOW_THREADS=1 python experiment.py --phase chain --kernels single burst5 decay-slow \
    --Ts 80 --seeds 11 22 33 44 55 --epochs 500 --jobs 6 --out results/chain
python analyze.py && python plot.py     # REPORT.md / write_dynamics.png / chain_kernels.png
```

kernel 定义：`single=[1]`、`burst3/5` 均匀、`decay-fast/slow ∝ 0.5^k / 0.85^k`（L2 归一）。Phase A 固定 B、核心按 V20 协议训练；Phase B 用 hybrid（B full / core K=5），`single` 为 V23 回归锚点。

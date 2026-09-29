# 第二十六轮：Budget-Matched Temporal Write Re-baseline

- 设计见 [PLAN.md](PLAN.md)（预注册），结果见 [REPORT.md](REPORT.md)（`analyze.py` 自动生成）。
- 问题：V25 只在 delay T80/task0（3 seeds）上发现 fixed kernel 的领先到 500ep 收缩到 ±4pp；V24 的 T80 四任务 +3.5pp 与 T160 +5.3pp 仍是 65ep 口径。本轮用**相同 500ep 预算**公平比较：**最终能力（H-cap）还是收敛速度（H-opt）？**
- 矩阵（只换 write kernel，其余与 V24 Phase A 一致；tasks 0–3、seeds 11/22/33/44/55、500 epochs）：

| Phase | T | Arms | Runs |
|---|---:|---|---:|
| A | 80 | single / burst5 / decay-slow | 60 |
| B | 160 | single / burst5 / decay-slow | 60 |
| C/D | — | adaptive bank（仅当 A 或 B 上 H-cap 成立） | 条件 |

## 主要结果

- **回归**：120/120 cells 的 `test@65` 与 V24 `delay_*` 同 seed **逐位一致**（max|Δ|=0）。
- **T=80：收敛速度效应**。single 0.954 / burst5 0.978（+2.4pp，4/5 同向）/ decay-slow 0.973（+1.9pp，4/5）；AUC 0.927→0.941，E95 中位 105→57.5/60。判定：burst5 落在 weak 带（2–3pp，不判 H-cap），decay-slow 判 H-opt。
- **T=160：能力效应**。single 0.843±0.025（E95 中位 453，仅 7/20 达到 95）vs burst5 **0.989±0.002**（+14.6pp，5/5 同向，4/4 tasks +8.4～+19.7pp）/ decay-slow 0.990±0.001（+14.7pp，5/5）；ΔAUC +19.3pp。**H-cap 成立**。注意 single 在 500ep 仍缓慢上升（Acc451-500 0.829 → Acc500 0.843），burst5/decay-slow 已饱和（0.987→0.989）——500ep 不是“充分收敛”证明，但同预算下差距 15pp。
- 结论：**短序列（T=80）主要是优化加速；长记忆距离（T=160）开始出现最终能力差异**，支持多时间尺度状态方向。按预注册，adaptive bank（C/D）与 continuous λ 现在有条件展开（建议先在 T=160、500ep 上跑 bank selector，再谈参数化比较）。

## 运行

```bash
# Phase 0 回归：train_rebase 前 65 epoch 三个 kernel 必须与 V24 delay_* 逐位一致
FLOW_THREADS=1 python verify_rebase.py

# Phase A / B（B 的 single/burst5 先入队；本机 T=160 建议 jobs 3–4）
FLOW_THREADS=1 python experiment.py --T 80  --kernels single burst5 decay-slow --tasks 0 1 2 3 --seeds 11 22 33 44 55 --epochs 500 --jobs 6 --out results/rebase_80
FLOW_THREADS=1 python experiment.py --T 160 --kernels single burst5 decay-slow --tasks 0 1 2 3 --seeds 11 22 33 44 55 --epochs 500 --jobs 3 --out results/rebase_160

python analyze.py   # REPORT.md / results/summary.json
```

注意：本轮只存 JSON、无检查点；`analyze.py` 同时打印全部 120 个 cell 的 V24@65 回归差与预算曲线。smoke：`--T 80 --kernels single burst5 --tasks 0 --seeds 11 --epochs 2 --out smoke`。

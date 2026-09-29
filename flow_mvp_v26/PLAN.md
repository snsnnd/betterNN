# 第二十六轮：Budget-Matched Temporal Write Re-baseline

## 0. 问题

V25 在 delay T80/task0（3 seeds）上发现：fixed kernel 的 65ep 优势到 500ep 收缩到 ±4pp，adaptive 与 fixed single 齐平。但该结论只覆盖 **task0、T=80、3 seeds**；V24 的 T=80 四任务 +3.5pp 与 T=160 +5.3pp 都还是 65ep 口径。本轮用**相同充分预算**公平比较：

\[
\boxed{\text{Temporal Write 提高最终能力（H-cap），还是主要提高收敛速度（H-opt）？}}
\]

此问题决定 V24–V25 这条线是继续（temporal lifetime 做成可学习机制）还是收缩到 Chain-select（那里 500ep 证据已成立）。**本轮不跑 adaptive bank / continuous λ**；只有 fixed 优势仍满足 H-cap 才开 C/D。

## 1. 不变量与协议

- Flow-v2、B fixed-disjoint（`E.make_fixed_B(α=0)`）、Route/Hold、数据、Adam lr=0.003、batch 128、core K=5、RNG 手法全部与 V24 Phase A 一致；**只换 write kernel**。
- kernels：`single`、`burst5`、`decay-slow`（V24 已证明代表瞬时/平铺/衰减三种结构）；等能量 `Σa²=1`，L=5。
- T∈{80,160}、tasks 0–3、500 epochs、seeds 11/22/33/44/55。
- **Phase 0 回归**：`single@T80/task0/seed11/65ep` 的 `final` 必须与 V24 `results/delay/delay_single_T80_t0_11.json` 逐位一致（同一 `train_fixed` 前 65 epoch；新增的中间 eval 不消费 RNG）。不通过则先修 harness 再跑网格。

## 2. 矩阵

| Phase | T | Tasks | Arms | Epochs | Seeds | Runs | 目的 |
|---|---:|---|---|---:|---|---:|---|
| A | 80 | 0/1/2/3 | single / burst5 / decay-slow | 500 | 11/22/33/44/55 | 60 | T80 四任务最终差距是否仍存在 |
| B | 160 | 0/1/2/3 | single / burst5 / decay-slow（single/burst5 优先入队） | 500 | 11/22/33/44/55 | 60 | V24 的 +5.3pp 是否只是慢收敛 |
| C/D | — | — | adaptive bank（kernel bank selector） | 500 | 同上 | 条件 | **仅当 A 或 B 上 H-cap 成立**才实现；否则不跑 |

## 3. 指标

- **能力**：`Acc500`（1024 test，最后 epoch）与 `Acc451-500`（val 在 epoch 451..500 每 epoch 评估的 50 点均值；避免单 epoch 振荡）。
- **优化速度**：`E90/E95/E98` = val 上第一次满足“该点及之后连续 2 次评估 ≥ 阈值”的 epoch（评估网格：1、前 450 每 5 epoch、末 50 每 epoch；未达到记 `None`，统计单列）。`AUC = trapz(val acc, 1..500)/500`。
- 学习曲线：val 在 25/50/65/100/200/300/500 的准确率；test@500。全部写入单条 JSON。

## 4. 预注册判定

对每对 kernel（先 `burst5−single`，`decay-slow−single` 同规则）与每个 T：先把每条 seed 的差在 4 个 task 上平均，再统计 5 seeds。

- **H-cap（最终能力）**：`mean ΔAcc500 ≥ +3pp` 且 **≥4/5 seeds 同向**；`Acc451-500` 用同一规则复核（两个指标都过才判 H-cap）。2–3pp 之间记为“弱差异（不判 H-cap）”。
- **H-opt（优化效率）**：`|mean ΔAcc500| < 2pp`，且 ≥4/5 seeds 同向地满足 `ΔAUC ≥ +1pp` 或 `E95 比值 ≤ 0.8`；若 single 到 500ep 未达 95 而对照 kernel 达到，按强烈优化效应计（单列）。
- **H-null（弱/无效应）**：其余。
- **结论映射**：
  1. T=80 H-cap → 与 V25 task0 相反，先核对口径（seeds/tasks）再解释；
  2. T=160 H-cap → 长记忆距离下 temporal kernel 改变最终能力，支持多时间尺度状态方向；
  3. 仅 H-opt → Temporal Write 在 delayed benchmark 上是优化加速，delayed 线降级，主证据收缩到 Chain-select；
  4. H-null → 同上降级。
- 只有在 A 或 B 上 H-cap 成立，才实现并跑 C/D（adaptive bank）与后续 continuous λ。

## 5. 边界

- 只覆盖 delay benchmark、T=80/160、B fixed；Chain-select 不重跑（V24 500ep 证据：0.777/0.913 → 0.971/0.999）。
- E90/E95/E98 与 Acc451-500 均在 val 上定义，不用测试集选模。
- 500 epochs 是共同预算上限，不是“充分收敛”证明；结论限于“同预算下的能力与效率”。
- 高准确区 3pp 已很显著；若只有 ~2pp 的稳定差，本轮不宣称 H-cap。
- 3 kernels 之外的 kernel（burst3/decay-fast）不跑，避免网格膨胀。

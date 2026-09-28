# 第二十四轮：Input Write Dynamics（等能量下“怎么随时间写入”值不值钱）

## 0. 背景与问题

V20–V23 已经排除了几件事：B 拓扑在任务侧无差异（V20）、B 解耦本身不是关键（V21）、core 的 5 步信用窗口足够（V22/V23）。剩下一个没人动过的结构假设：

> **当前是“单次瞬时写入”**：信息只在 t=0/1 通过 `z_t = B x_t` 一次倒进状态，之后完全交给 W/Route/Hold 保存几十步。这会不会本身就是脆弱性来源？

V24 不改模型结构、不加参数，只把写入改成**有时间结构的内部 write trace**：

\[
z_t=\sum_{k=0}^{L-1} a_k\,B\,x_{t-k},\qquad \sum_k a_k^2=1 .
\]

**外输入仍然只出现一次**；重复的是模型内部对同一条输入的因果 FIR trace，不是把答案重新喂给模型。`Σa_k²=1` 保证总写入 L2 能量与 single 相同（否则 burst 变好可能只是“灌了五倍的水”）。

## 1. Kernels（固定，不训练）

| arm | 系数 |
|---|---|
| `single` | [1]（= 现有行为，跳过卷积，保持逐位回归） |
| `burst3` | [1/√3, 1/√3, 1/√3] |
| `burst5` | [1/√5]×5 |
| `decay-fast` | \(a_k\propto 0.5^k, k=0..4\)，L2 归一 |
| `decay-slow` | \(a_k\propto 0.85^k, k=0..4\)，L2 归一 |

## 2. Phase A：延迟任务 × 固定 B（主实验）

- 任务：V20 的 A'B'C'D'，但序列长度 **T∈{20,40,80,160}**（输入 t=0/1、规则 t≥4、末尾读出）。
- B 固定 = V20 `fixed-disjoint`（overlap α=0，`‖B_c‖₂=5.6`）；W/Route/Hold/Readout 按 V20 协议训练：5 步截断、Adam lr=0.003、batch 128、65 epochs、固定最后 epoch。
- 网格：5 kernels × 4 tasks × 4 T × 5 seeds = **400 条单任务流**。
- 指标：
  - `final`：1024 样本测试集两输出全对率；
  - **response retention**：反事实响应 `‖Δh_t‖`（只给通道 A / 只给通道 B 相对 zero-input），参考点 t=5（所有 kernel 注入完毕），记录 t=0.25T、0.5T、T−1 的保留率，以及 [5,T−1] 的 log 线性衰减率（lifetime = −1/slope）；训练后为主，init 为次。
- Phase 0 回归：`single@T20` 必须与 V20 fixed-disjoint singles **逐位一致**（已用 seed11/task0 抽查：0.98828125 完全相同；全量核验写进 analyze）。

## 3. Phase B：Chain-select × K=5（条件实验，H1 成立才展开）

- 用 V23 的 Chain-select（T=80、xor/xorsw），保持 **learnable B + hybrid**（core K=5、B full）与 500 epochs 不变；
- 只改 write kernel：`single`、`burst5`、`decay-slow`（3 个）× 2 tasks × 5 seeds = 30 流；
- `single` 是 V23 同 cell 的回归锚点；
- 回答：**V23 K5 的部分失败（soft-selection、可达性不足）是否来自“一次性写入太脆弱”？**

## 4. 预注册假设

- **H1（写入时间结构有效）**：Phase A 中至少一个非 single kernel 在 **T=80** 的 4 task × 5 seeds 平均 `final` 相对 single **≥ +5pp**，或 t=0.5T 的 retention 相对提升 **≥20%**。
- **H1-null**：若所有非 single kernel 在 T=80/160 的 |ΔAcc| ≤2pp 且 retention 差 ≤10% → 单次写入不是当前结构瓶颈，**冻结 V24B/C（adaptive scheduler）方向**。
- **H2（优势随 T 不消失）**：`ΔAcc(T=160) ≥ ΔAcc(T=20)`；同时报告 T=40/80 的走势。若只在 T=20 有优势、T≥80 消失，则警惕窗口对齐假象。
- **H3（Phase B）**：Chain-select T=80 上最优 kernel 相对 `single` **≥ +5pp 且 ≥0.90**；不成立则 V23 的失败与写入时间结构无关。
- **H4（exploratory 机制）**：retention 提升与 acc 提升同向（kernel 通过减慢响应衰减来帮助长 T）。

只有 H1 成立才进入 V24B（可学习 write gate `w_t`，只看 `(x_t, meta_t)`）与 V24C（可学习 `λ_t`）；否则该方向整体冻结。

## 5. 边界与风险

- Phase A 固定 B 以隔离“写到哪里”；kernel 固定，验证的是“时间结构这个自由度是否有价值”，不是调度器本身。
- 不扫描幅度（`Σa²=1` 固定能量）；不与 V20 的 `s_W` 交互（已零结果）。
- Phase B 仍用 V23 的 hybrid（B full / core K=5），`single` 作为回归锚点；结论只针对该任务族。
- 65 epochs 对所有 T 一致，但长 T 任务本身更难；acc 绝对值不是重点，同 T 内的 kernel 对比与 retention 才是。

## 6. Addendum（Phase A 运行后、分析前的诊断补充）

- `ret_half/ret_end` 的“相对 single”用**几何均值比（mean log-ratio）**，避免小分母离群值把百分比拉爆；H1 的 retention 判据相应改为“几何均值比 ≥ +20%”。
- 追加 **linear probe**（前 512 样本拟合、后 512 评估，两输出平均）测“早期信息到 t 还可解码吗”（只用测试种子数据，不参与训练）。该指标是分析阶段补充的诊断，写入 REPORT 的 §2.5；H4 仍按预注册用 `||Δh||` retention 判定，probe 相关作为 exploratory 附加报告。
- 这两项都在看到 Phase A 的 mean 结果之后加入，不改变任何训练协议或网格。

## 7. Phase C（V24B）：可学习 Write Scheduler（`w_t`）

Phase A/B 证明了固定 kernel 有效、但**最优 kernel 依任务而变**（xor→burst5、xorsw→decay-slow）。Phase C 让网络自己决定“什么时候写、写多强”。

### 7.1 结构（不改 W/Route/Hold/B 的结构）

\[
e_t=w_t\odot x_t,\qquad z_t=\sum_{k=0}^{L-1}a_k\,(e_{t-k}B),\qquad w_t=2\,\sigma\!\big(\mathrm{MLP}(x_t,\mathrm{meta}_t)\big)
\]

- controller：`Linear(9→16)+Tanh+Linear(16→1)`，末层零初始化 → 初始 `w≡1`，即**训练起点与对应固定 kernel 完全一致**（forward 逐值相等）；
- **controller 只看 `(x_t, meta_t)`，不看 `h_t`**（保持并行友好，不引入新的强循环依赖）；
- 第一版固定 `a_k`（沿用 Phase A/B 的 kernel），controller 只决定“什么信息值得写”；`λ_t`/`α_t` 留 V24C。

### 7.2 信用

`∇w` 在纯 5 步截断下同样为 0（事件都在窗口外），因此 **controller 与 B 走 full BPTT（hybrid 双通道），core 仍 K=5 截断**；controller 只有 177 个参数。

### 7.3 网格

| 组 | 任务 | 臂 | seeds | runs |
|---|---|---|---|---|
| Q1/Q2 | Chain-select T=80、xor/xorsw | kernels {single, burst5, decay-slow} + scheduler | 11/22/33/44/55 | 30 |
| Q3 | Chain-select T=160、xor | fixed {single, burst5} + burst5+scheduler | 11/22/33 | 9 |
| 回归 | 初始化一致性 | scheduler 开启/关闭 vs 固定 arm | — | Phase 0 |

延迟任务不跑：那里所有输入都重要、没有 distractor，scheduler 没有决策空间。

### 7.4 指标

- final acc（1024 test）与 best-val reach；对照 V24 Phase B 的 fixed-kernel 结果；
- **事件级 `w`**：τA / τc1 / τdistractor / τB / τc2 的均值，以及全步均值（检查是否“全开”退化）。

### 7.5 预注册假设

- **H5（能力）**：T=80 上存在**同一个** scheduler 配置同时满足 `xor ≥0.95`、`xorsw ≥0.95`，且两个任务都相对 fixed `single` ≥ +5pp。
- **H6（重要性）**：至少 2/6 个 (task, kernel) cell 满足 `E[w|distractor] ≤ E[w|important] − 0.20`（important 取 A/c1/B/c2 均值的最小值，按 seed 平均），且全步均值未坍缩到端点（∈(0.02,1.98)）。
- **H7（长 T）**：T=160 xor：`burst5+sched` 不劣于 `burst5 fixed`，且相对 fixed `single` ≥ +5pp。
- 若 H5 成立但 H6 不成立（accuracy 好但 `w≈1` 全开）→ scheduler 只是隐式正则，不算“学会重要性”；若 H5 不成立 → 单靠输入权重不足以复现最优固定 kernel，需要 V24C 的动态 `λ_t`。

# 第二十三轮：Credit-Stress Benchmark（先造真正需要长程 core 信用的任务）

## 0. 背景与目标

V22 在 A'B'C'D'（T=20、规则持续可见、输入只在 t=0/1）上证明：给 core 打开 20 步 Full 信用不改变遗忘（G=0.17pp）。但这不能推出"core 永远不需要长程信用"，因为旧任务的结构恰好允许最后 5 步完成计算：

- 规则 R 在 t≥4 持续出现在 meta 里，最后窗口直接可见；
- 输入一次性写入（B 有 Full credit），core 只需保持到末尾。

**V23 不改模型，只造任务**：构造一个"瞬态事件 + 长间隔 + 中途干扰 + 条件选择"的任务族，先回答是否存在某个 (T, K)：

\[
\mathrm{acc}(K=5)\ll \mathrm{acc}(\text{Full}).
\]

- 若存在 → 抓到 Flow 的 credit-assignment hard case，V24 再研究局部/并行长程信用；
- 若 T 扫到 160 仍不存在 → 强结论：**Flow 的状态系统只需要短窗口局部信用就能形成长期行为**（强化 Round 5）。

## 1. 任务族 Chain-select（唯一改动）

序列长度 T；事件时刻取 `round(f·T)`，f = (.05, .21, .39, .58, .76)：

| 事件 | 通道 | 值 | 说明 |
|---|---|---|---|
| A（早期输入） | x ch0 脉冲 | ±1 | 最终可能被选中的值 |
| c1（早期 context） | x ch0 脉冲（transient）或 meta 持续（sustained 对照） | ±2 | 控制位 1 |
| distractor | x ch0 脉冲 | N(0,1) | 干扰项 |
| B（后期输入） | x ch1 脉冲 | ±1 | 另一个候选值 |
| c2（后期 gate） | x ch1 脉冲（transient）或 meta 持续（sustained） | ±2 | 控制位 2 |

- meta：ch2 = 归一化时间；ch3 = 1 当 t ≥ τ_c2（"作答阶段"）；sustained 变体额外用 ch0=1、ch1=c1（t≥τ_c1）与 ch4=c2（t≥τ_c2）。
- 读出只在最后一步（`headX` 读 role2、`headY` 读 role3）。
- 任务（两个镜像，防单任务巧合）：
  - `xor`：`selected = A if c1==c2 else B`；`y = [sign(selected)>0, 1(c1>0)]`
  - `xorsw`：`selected = B if c1==c2 else A`；`y = [sign(selected)>0, 1(c2>0)]`
- 两个输出全对才算正确（与旧 benchmark 一致）；chance ≈ 25%。

**为什么这个任务可能暴露 core 信用窗口**：c1/c2 是瞬态脉冲，任务要求在 t≈0.76T 处把"同/异"与存下来的 A/B 组合出被选值。K=5 的窗口（t=T−5..T−1）既不包含任何事件，也不包含 c2；core 在脉冲时刻的行为只能通过共享参数泛化得到。A'B'C'D' 里规则持续可见、没有这种"脉冲必须被 core 在窗口外处理"的结构。

## 2. 信用协议（与 V22 同源）

- **B 恒可训练、恒 Full**（overlap 初始化、`‖B_eff‖₂=5.6`、写入域 role0∪role1）；
- core = `{W, route, hold}` 使用截断窗口 `detach_period = K`；readout 与 B 走 Full 通道（只在最后一步使用，数值上等价）；
- K ∈ {5, 10, 20, Full}；Full = `detach_window=False`（等价于 K≥T）；
- `K<Full` 时按 V22 双通道：同一快照跑截断通道（core 梯度）+ 完整通道（B/readout 梯度），双 optimizer、分别 clip；
- `K=Full` 时单次完整 backward + 同样的分别 clip（省一次前向）；
- 训练：Adam lr=0.003、batch 128、**500 epochs**、512 训练/256 验证/1024 测试、固定最后 epoch（不用测试集选模）；每 25 epoch 记 val 曲线。

## 3. 网格（预注册）

| 组 | 任务 | 变体 | T | K | seeds | runs |
|---|---|---|---|---|---|---|
| 主 | xor, xorsw | transient | 20 | 5, 10, Full | 11/22/33/44/55 | 30 |
| 主 | xor, xorsw | transient | 40 | 5, 10, 20, Full | 同上 | 40 |
| 主 | xor, xorsw | transient | 80 | 5, 10, 20, Full | 同上 | 40 |
| 缩放 | xor | transient | 160 | 5, 20, Full | 11/22/33 | 9 |
| 对照 | xor, xorsw | sustained | 80 | 5, 20, Full | 11/22/33/44/55 | 30 |

共 149 条单任务流。T=160 单流约 20–45 min（RSS≈1.1GB），预算限制下用同一 500 epochs 但只跑 3 seeds。

## 4. 预注册假设

- **H0（可学性校准）**：transient、T∈{20,80}、Full 的两个任务均值 ≥0.90，否则任务无效（只报告为"架构/优化边界"，不得用于信用结论）。
- **H1（T=80 出现 breakpoint）**：transient、每个任务：`mean(Full) ≥ 0.90` 且 `mean(Full) − mean(K=5) ≥ 0.15`，且 5 seeds 中 ≥4 个方向一致。两个任务都过 → "hard case found"；一个过 → partial。
- **H2（窗口内容决定成败）**：T=80：`K=20 − K=5 ≥ 0.15`（K=20 窗口 t=60..79 覆盖 c2@t=61）；`K=10 − K=5 ≤ 0.05`（K=10 窗口 t=70..79 无事件，应与 K=5 同类）。
- **H3（T 缩放）**：transient、xor：`gap(T=80) ≥ gap(T=20) + 0.10`（gap = Full − K5）。
- **H4（变体对照）**：T=80：transient 的 gap ≥0.15 时，sustained 的 gap ≤0.05（规则持续可见时最后窗口足以学会选择）。
- **H5（T=160 二次缩放）**：T=160 时 `K=20` 也失败（gap ≥0.15），即窗口必须覆盖至少一个事件；Full 仍 ≥0.90。
- **H6（保持/组合分解，exploratory）**：K=5 失败主要落在 `y0`（选择）而不是 `y1`（context 位读出）——即"保持没问题，跨时间组合才是瓶颈"。

## 5. 判定与结论分支

- H1 成立 → V23 定义了 Flow 的 credit-stress 基准（T=80 起），V24 研究 eligibility/local credit 的目标明确；
- H1 不成立且 T=160 也无 gap → **强正向结论**：core 的 5 步 TBPTT 在瞬态长程任务上也足够，Flow 的长期行为不依赖 core 长程信用；Flow-v3 只需要解决 B 的跨窗口写入信用。

## 6. 边界与风险

- 只改任务生成器；模型/协议与 V21/V22 逐位同源（Flow 类照抄 v22）。
- Full 校准本身可能失败（尤其 T=160）；H0/H5 的 Full 门槛是硬检查，失败则记为边界而不是信用效应。
- T=160 单流约 33 min（500 epochs，RSS≈1.1GB），并行度受内存限制（T=160 最多 4 进程）。
- 两个任务均为合成二分类，结论限于该任务族；所有数字固定最后 epoch。

## 7. Addendum（主网格运行期间发现，分析前的诊断修正）

主网格运行到约 2/3 时观察到：xor 任务上存在**"软选择"平台与周期振荡**——多条流（含 Full）在 val 0.75–0.85 平台长期停留，或先到 0.98–1.0 再在最后几十 epoch 崩回 0.5；固定最后 epoch 会把"曾经学会"的流记为失败。为把**可达性**（能否找到解）与**稳定性**（能否保持到训练结束）分开：

- 主指标保持预注册的 `final` 不变；
- 追加诊断 `best_val = max(val curve)`（曲线每 25 epoch 记录；只用验证集，不做测试集选模），报表同时给出 `best-val reach`（≥0.9 的 seeds 数）；
- H0/H1 同时按两种口径报告；若结论冲突，以 final 为准，并明确标注为"可达但不可稳定保持"。

该修正在看到主网格约 2/3 结果后加入，属于诊断性补充（不改协议、不改训练、不换模型），并在 REPORT 中如实标注。

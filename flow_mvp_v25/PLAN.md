# 第二十五轮（V24D）：Temporal Strategy 的可发现性（Phase 0 Optimization Audit）

## 0. 背景与问题

V24C 的观察（**已观察，不是解释**）：

- Chain-select T=80 上 kernel bank selector 成立：xor 0.998（超 best fixed burst5 0.971）、xorsw 1.000（=best fixed decay-slow 0.999）；HC1/HC2 通过（HC2 由 xorsw 单项满足）。
- xor 学到的是**全局长时间 kernel 策略**（几乎所有事件铺开，distractor L_eff=1.66 高于重要事件均值 1.30），不能作为“按重要性分配寿命”的证据。
- xorsw 是目前唯一的**逐事件寿命选择**证据：只有 c2 得到长 kernel（L_imp/L_d=2.51，4/5 seeds）。
- 延迟任务上：fixed kernel 已知有更优解（T80 +3.5pp、T160 +5.3pp），但 adaptive selector 未发现它（T80 task0 0.734 vs best fixed 0.898；T160 0.572 vs 0.606），α 停在 single。

因此当前核心未决问题不是“bank 还是连续 λ”，而是：

\[
\boxed{\text{失败来自 representational limitation，还是 optimization / exploration limitation？}}
\]

这两个不拆开，V24D 主实验（连续 `λ_t`）即使成功也无法归因。本轮 Phase 0 只回答“为什么 selector 不动”，**不**跑主实验。

## 1. 不变量与协议

- 任务：延迟 A'B'C'D' 的 **task0、T=80**（延迟任务中 adaptive 与 best fixed 差距最大）；数据生成与种子协议同 V24（`seed*10000 + task*100(+1/+10)`）。
- B 固定 = V20 `fixed-disjoint`（`E.make_fixed_B(m, seed, 0.0)`）。
- core 仍 5 步截断；scheduler 与 B 同走 hybrid full BPTT（V24B/C 协议），core K=5、Adam lr=0.003、batch 128。
- 等能量：所有 kernel/混合均 L2 归一（`Σa²=1`），kernel 长度 L=5。
- seeds：screening 用 **11/22/33**；若判定落在阈值附近（见 §4 风险）再补 44/55。
- 回归锚点：
  - `A1` 必须与 V24 `results/bank_delay/bank_delay_T80_t0_{seed}.json` **逐位一致**（同代码路径、同 RNG）；
  - 65-epoch fixed 对照直接读 V24 `results/delay/delay_*_T80_t0_{seed}.json`；
  - `F*` 500-epoch fixed 用同一 `train_fixed` 重跑，作为 `A4/B3` 的同预算对照。

## 2. Phase 0 矩阵

| arm | 参数化 | 初始化 | epochs | 目的 |
|---|---|---|---|---|
| A1 | bank（per-event α，MLP） | single（[4,0,0,0,0]） | 65 | V24C 复现锚点 |
| A2 | bank | burst5（[0,0,4,0,0]） | 65 | 好核初始化能否保持 |
| A3 | bank | decay-slow（[0,0,0,0,4]） | 65 | 同上（T80 最优 fixed） |
| A4 | bank | single | 500 | 65→500 epochs 是否只是协同适应慢 |
| B1 | shared λ（1 个标量） | λ0=0.05（≈single） | 65 | 去掉条件化，一维连续参数是否有梯度信号 |
| B2 | shared λ | λ0=0.6（中间） | 65 | 中初始化的行为 |
| B3 | shared λ | λ0=0.05 | 500 | 预算对照（shared λ 版） |
| F1 | fixed single | — | 500 | A4/B3 同预算对照 |
| F2 | fixed burst5 | — | 500 | 同上 |
| F3 | fixed decay-slow | — | 500 | 同上 |

3 seeds × 10 arms = 30 runs。`A1` 通过的逐位回归同时验证 harness 正确性。

## 3. 诊断指标

1. **参数位移**（回答“有没有动”）：bank 用逐事件 `‖α_end−α_init‖₁`（A/B 两事件平均）；shared λ 用 `|λ_end−λ_init|`。同时在 curve 里每 25 epochs 存一次 λ/α 画像。
2. **性能**：final（1024 test）与 best-val（V23 教训：必须同时给可达性，避免把震荡读成信用/优化问题）。
3. **初始梯度（full BPTT）**：同一 batch 网格（4×128）上记录
   - bank：对 softmax logits（= 初始 bias）的梯度 5 维均值/标准差（初始 hidden=0，bias 梯度就是“把概率推向 kernel i”的信号）；
   - shared λ：`∂L/∂s` 的均值/标准差，并换算 `∂L/∂λ = ∂L/∂s·σ'(s)`；
   - 同时记录截断通道梯度：detach 在每个 5 步边界，损失只连到最后 5 步，delay 的写入（t≤5）预期截断梯度恒为 0（协议核验）；scheduler 的真实梯度只看 full 通道，其量级可能很小，记录数值与方向。
   判定口径：`|mean|` 相对 batch 间 std 是否可忽略（flat vs conflict）。
4. **条件 loss landscape**：训练前后冻结全部参数，只扫描 kernel：
   - bank：`K(γ)=normalize((1−γ)K_single+γK_j)`，`j∈{burst5, decay-slow}`，`γ∈[0,1]` 21 点；
   - shared λ：`λ∈[0,0.99]` 21 点（`a_τ∝λ^τ`，L2 归一）；
   - 记录 val BCE loss 与 acc。**边界**：这是条件 landscape（core 已按自身策略训练），只作局部几何证据；联合可达性由 fixed-kernel arms（V24 Phase A + F*）提供，不用它下“不存在好解”的结论。

## 4. 预注册判定（Phase 0）

- **Q1（init basin）**：
  - “停留”：`mean_e ‖α_end−α_init‖₁ < 0.2` 且 `final ≥ best fixed(65ep) − 2pp` 且 `final ≥ A1 + 2pp`；
  - “塌回”：`‖α_end−α_single‖₁ < 0.2`（one-hot）或 `final ≤ best fixed − 5pp`；
  - 其余记“中间/混合”。
- **Q2（shared λ）**：
  - “会动”：`|λ_end−λ0| ≥ 0.2` 且 `final ≥ A1 + 2pp`；
  - “不动”：`|λ_end−λ0| ≤ 0.05` 且 `|Δacc| ≤ 1pp`；
  - 无条件报告初始化 `∂L/∂λ` 的符号与 batch 间 std。
- **Q3（预算）**：“预算问题” = `A4 ≥ A1 + 2pp` 且 `A4 ≥ F1 + 2pp`（且 3 seeds 中 ≥2 同向）；“非预算” = `A4 ≤ A1 + 1pp`；B3 同法对照 B1。
- **决策映射**（可组合，不互斥）：
  1. A2/A3 停留 + B 不动 → **可发现性/探索**问题主因：Phase 2 优先（探索/退火/continuation）；
  2. B 动但 A1/A2 不动 → **条件化参数化**（`x_t,meta_t→λ_t` 的 MLP）稀释了梯度：Phase 1 优先（连续 per-event λ 的优化几何）；
  3. 只有 A4/B3 改善 → **预算/协同适应慢**：重设 epochs/退火 schedule 后重审，不换表示；
  4. A2/A3 塌回 single 或明显劣于对应 fixed → **联合训练破坏好解**：先查 hybrid 两遍协议与 core-scheduler 耦合，暂停 Phase 1/2。
- **风险**：3 seeds 下阈值附近不做强结论；单点 landscape min 不作为机制证据。

## 5. 边界

- 只跑 task0、T=80；结论不外推到 T=160 与 Chain-select。
- shared λ 只覆盖指数族（single→decay family→burst5），**不包含 burst3**；λ→1 在有限窗下=burst5，不是无限寿命。
- bank 的 γ 扫描是系数空间线性插值再归一，不是 α 空间插值。
- `A1` 与 `F1`（65ep）读/重跑 V24 协议，不得改 lr/batch/data/RNG。
- 本轮不实现 Phase 1/2（见 §6），避免看到 Phase 0 结果后改判据。

## 6. Phase 1/2（预注册，本轮不实现）

- **Phase 1（Parameterization Test）**：per-event 连续 `λ_t=σ(MLP([x_t,meta_t]))`（9→16→1），与 bank selector 同 hybrid 协议、**无探索**；问题写成“continuous λ 是否提供更好的 optimization geometry”，不是“表达力更强”（其可达集不包含 burst3，未必更大）。
- **Phase 2（Exploration Test）**：`2×2`：{bank, 连续 λ}×{无探索, 匹配探索}。
  - 匹配定义（先做校准）：训练时对 scheduler 隐参数加高斯噪声，噪声尺度按**诱导 kernel 系数扰动的 RMS** `E‖Δk‖₂` 匹配（而不是原始噪声 std），并在 REPORT 里给出两参数化的校准曲线（噪声尺度→诱导扰动）。
  - 估计 parameterization effect、exploration effect 与交互。
- Phase 1/2 的具体判定阈值在 Phase 0 报告后、跑 Phase 1/2 前另写 Addendum 预注册（不允许看到结果再改）。

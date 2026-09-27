# 第二十轮：B 输入拓扑 × 底网动力学

## 0. 背景

V19 把“时间依赖可压缩”推进到 GPU wall-clock 后得到负工程结论：structured affine scan 在 N=256、eager PyTorch 下没有加速价值，且长乘积数值不稳定；据此并行线冻结在 V19。本轮回到主线问题：

> **水从哪里进入（B 的支撑与符号），以及入口如何与河床动力学（W 初始尺度、路由/保持）交互。**

不改 Flow-v2 架构（W + Route + Hold + Readout），只改 B 的拓扑/归一化与 W 初始尺度。对象为 v15 检查点的构建协议与训练协议。

## 1. 主族：B topology（保持角色写入权限）

v15 结构：通道 0 只写 role0（来源 A），通道 1 只写 role1（来源 B）。主族保持该权限，只改每通道支撑大小 k：

| 拓扑 | k | 支撑定义 |
|---|---:|---|
| single-random | 1 | 该 role 内 1 个随机节点 |
| single-central | 1 | `i* = argmax_i ‖(W⊙mask)_{i,:}‖₂`（初始出边权重范数最大的节点） |
| multi-separate-8 | 8 | 该 role 内 8 个随机节点 |
| multi-separate-16 | 16 | 该 role 内 16 个随机节点 |
| distributed | 64 | 该 role 全部节点 |

- 每 topology 的支撑与符号由独立 RNG 生成（seed 派生），**不影响** Hold/Route/Head/W 的初始化流。
- 实现顺序：先按 v15 正常构建整个模型（含 W/mask/B/Hold/Route/Head），再覆盖 B。B 覆盖不得改变任何其它参数的 RNG 消耗（回归可比性）。

## 2. shared-pool overlap family（单独实验，不与主族混比）

放开写入权限：两个通道都可写 `role0 ∪ role1`。每通道固定 `k=16`，且严格 **role0 内 8 个 + role1 内 8 个**。

每 role 内两通道支撑交集大小为 0 / 4 / 8 ⇒ 总共享节点数 0 / 8 / 16：

| α_overlap | 总共享节点 | 每 role 交集 |
|---:|---:|---:|
| 0 | 0 | 0（matched-disjoint baseline） |
| 0.5 | 8 | 4 |
| 1 | 16 | 8 |

- 共享节点上按“半数同号、半数反号”分配符号，使 `|B₀ᵀB₁| / β² ≤ 0.05`（近似正交），把干扰归因于**共享神经元支撑**而不是输入向量相关。
- α=0 是该族的 matched-disjoint baseline，禁止拿主族 multi-separate 当 α=0 对照。

## 3. 输入预算（统一 L2 能量 + 饱和诊断）

- 每通道固定 `‖B_c‖₂ = β = 0.7·√64 = 5.6`（与 v15 期望尺度一致）。
- k 个节点：`|B_{ci}| = β/√k`，符号用固定种子的 Rademacher：单点 5.6、k=8: 1.98、k=16: 1.4、k=64: 0.7。
- 必须记录 tanh 饱和指标（t∈{0,1}，输入发生的步）：
  - `sat_prob = P(|x_t @ B| > 2)`（输入诱发的 pre-tanh 幅度）；
  - `mean_tanh_grad = E[1 − tanh²(pre_tanh)]`（全候选状态，含循环项）。
  否则 single-k1 若变差无法区分“支撑集中”与“单节点 5.6 直接饱和”。

## 4. 动力学轴（符号修正）

代码里是 `‖W₀‖₂ = 0.9`（`0.9·w/‖w‖₂`），不是谱半径。记：

- 主实验：`s_W = ‖W₀‖₂ = 0.9`；
- Phase C 扫描：`s_W ∈ {0.7, 0.9, 1.1}`。

Phase C 预注册两个端点拓扑：`single-random (k=1)` vs `distributed (k=64)`，**不做 post-hoc 选择**。训练过程中 W 不做 spectral 约束（与 v15 相同）。

## 5. 指标

1. **瞬时与有限时间动力学**（解析 `J_t = S_t + C_t`，v19 已验证）：
   - `ρ_eff = E_t[ρ(J_t)]`（瞬时谱半径均值）；
   - `γ_H = (1/H)·log‖J_{t+H−1}···J_t‖₂`，H ∈ {4, 8}（非正规有限时间放大）。
2. **state lifetime**：固定 operating point 的 Route/Hold（open-loop，冻结该点观测到的 `a_t, g_t`），对状态加小扰动 δh 传播：
   - 半衰期 `τ1/2`；
   - 最大瞬态放大 `G_max = max_k ‖δh_{t+k}‖/‖δh_t‖`（区分快速衰减 vs 先放大后衰减）。
   主报告 t*=8（规则到达后），并记录 t 扫描曲线。
3. **可达性**（双口径）：
   - 正式任务轨迹：边界状态协方差 PCA 有效秩 `r_PR = (tr C)²/tr(C²)`；
   - **独立 probe（不训练）**：`x_t ~ 小随机`（多点输入，meta 固定），测 `∂h_T/∂x_{1:T}` 的有效秩、奇异谱、σmax、input→state gain。避免把“任务只有 t=0/1 两个标量输入”误读为网络不可达。
4. **任务侧**：单任务可学性 + A'B'C'D' CL 的 acquisition / final_mean / forgetting（矩阵）。

## 6. 训练与评测协议

- A'B'C'D'，序列 20 步（外推 40 步）；train/val/test = 512/256/1024。
- 5 seeds（11/22/33/44/55），65 轮/阶段，5 步截断 BPTT，Adam lr=0.003，batch 128，固定最后 epoch。
- CL：`r ∈ {0, 12.5%}`，3 个任务顺序（o0/o1/o2），每任务回放缓冲 32 样本；主判定按合并顺序（与 v11 一致）。
- 单任务门槛：scratch ≥99%；不满足先查优化，不解读拓扑。
- 参考行：v15 原 B（Gaussian、期望范数同为 β）结果作为无干预参照；本轮 distributed 使用 Rademacher 归一化版本，两者不可逐位混比，只作量级对照。

## 7. 阶段

- **Phase A**：主族 5 topology（single-random / single-central / 8 / 16 / 64），统一 dynamics + reachability + 单任务 + CL。
- **Phase B**：shared-pool overlap α ∈ {0, 0.5, 1}，同指标。
- **Phase C**：`s_W ∈ {0.7, 0.9, 1.1}` × {single-random, distributed}，测 B×动力学交互。

## 8. 预注册假设与判定

- **H1（主族）**：等输入能量下，multi-separate（k=8/16）相比 single-random **降低 r=0 遗忘**。
  - 门槛：`ΔForgetting ≤ −3pp` 且 ≥4/5 seeds 方向一致；
  - r=12.5% 为次要：final 下降不超过 1pp（playback 优势不倒退、无副作用）。
- **H2（较弱）**：distributed 相比 single-random：`reachability ↑ ≥10%` 且 `τ1/2 ↓ ≥20%`，≥4/5 seeds 同方向。
- **H3（overlap → 冲突 → 遗忘）**：α_overlap ↑ ⇒ 任务梯度冲突 ↑。
  - 主指标：`Δmean cos ≤ −0.10` 或负 cosine pair 比例 `+≥20pp`，且 **W / Route / Hold 至少两组方向一致**（梯度按 v13 的截断梯度口径测）；
  - 辅助：forgetting 增加 ≥3pp。
- 判定失败/方向相反时：按“支撑隔离不是主因”记录，不调参续跑。

## 9. 输出

- `results/` 分阶段 JSON + 检查点（5 seeds × 每流阶段）；
- `analyze.py` 全量重载核验 + 汇总 + 自动写 `REPORT.md`（含判定栏）；
- `plot.py`：`input_topology.png`（accuracy/forgetting vs topology）、`dynamics.png`（ρ_eff/γ_H/τ1/2/G_max）、`overlap_gradient.png`（α vs cos/negative fraction）、`sW_interaction.png`（s_W × topology）。

## 10. 边界与风险

- 只改 B 与 W 初始尺度，不引入新门/新模块；不训练 probe。
- 支撑随机性：每 seed 独立生成并记录（可复现）；central 依赖初始 W。
- tanh 饱和可能主导 single-k1 的差异，必须与 sat_prob 一起解释。
- Phase C 的 s_W 只在初始化设定；训练中 W 自由更新，交互效应按“初始尺度”归因。
- 并行/solver 线冻结在 V19，本轮不做任何 scan/计时工作。

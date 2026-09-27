# 第二十一轮：Adaptive Input Decoupling（B 能否自己学会把不同信息写进不同状态子空间）

## 0. 背景与问题

V20 发现：等输入能量下改变固定 B 的空间支撑，任务侧几乎不变（H1 否），但 overlap 族显著变差（r=12.5% final 0.91–0.94 vs 主族 0.97–0.99），且 W/Hold 的梯度冲突上升（Δcos ≈ −0.11～−0.22）。机制解释是**状态资源竞争**：两个输入源写同一批状态自由度时，旧任务依赖的通路被新任务的 B 梯度直接改写。

V21 回答三个层级不同的问题：

1. 固定解耦有没有用？（V20 replication）
2. 网络自己会不会学会解耦？
3. 显式解耦正则是否额外减少梯度冲突与遗忘？

目标是把机制链 `B overlap ↓ ⇒ state overlap ↓ ⇒ gradient conflict ↓ ⇒ forgetting ↓` 变成可检验的预注册假设。

## 1. 写入域与归一化（收紧版）

- **learnable 域只允许 `role0 ∪ role1`（128 维）**：`B ∈ R^{2×128}`，global 列 0..127；role2/3（readout role）恒为 0，禁止输入旁路动力学直接进读出区。
- **能量恒定**：训练参数为 `B_raw [2,128]`；forward 使用
  `B_eff = β · B_raw / (‖B_raw‖₂ + ε)`，`β = 0.7√64 = 5.6`（每通道独立归一）。
  所有 penalty 与指标一律作用在 `B_eff` 上，绝不作用在 raw 上。
- 固定臂（fixed-overlap / fixed-disjoint）沿用 V20 的 `overlap_a1 / a0` 构造（同样只在 role0∪role1），等价于 `B_raw` 冻结且已归一。
- 初始化：learnable 默认从 **fixed-overlap 的 B** 出发（V21C 另做 matched random-init 对照）。

## 1.5 B 的梯度窗口（必须解决的 v13 陷阱）

**事实（已实测）**：任务输入只在 t=0/1，而训练在 t=5/10/15 处 `h.detach()`，因此标准 5 步截断下 `B_raw` 的梯度**恒为 0**（实测 ‖grad‖=0.0），与 v13 发现的 Write 门完全同源；全 BPTT 下非零（0.0029）。若直接用现有训练协议，`learnable` arm 的 B 永远不会更新。

**解决（预注册）**：V21 主实验使用 **严格 hybrid 信用分配**（论文口径）：

- 同一 batch、**同一参数快照**下计算两组梯度，之后才更新：`g_core = ∇θ_core L_trunc`、`g_B = ∇B_raw (L_full + λR_B)`；
- 两个独立 optimizer：`Opt_core = Adam(W, Route, Hold, HeadX/Y; lr=0.003)` 与 `Opt_B = Adam(B_raw; lr=0.003)`；
- 两次 `clip_grad_norm_(·, 1)` **分别**作用于 core 与 B，避免 B 梯度改变 core 的 clip 比例；
- core 的训练规则与 V20 逐位一致，因此 fixed 臂可直接复用 V20（只做 checkpoint 回归核验）；
- 硬检查（Phase 0）：`max|g_core^hybrid − g_core^V20| < 1e-6`、`|∇B|_trunc = 0`、`|∇B|_hybrid > 0`。

模式（`--b-grad`）：`hybrid` = 主协议；`full` = Phase D 协议敏感性（4 arms × 5 seeds × o0 × r∈{0,12.5%} = 40 流，全部全 BPTT，不进入主 H 统计）；`trunc` = 负控（5 seeds，仅验证 `max|∇B| < 1e-12` 且 `D_B = 0`，不跑全量）。

**penalty 口径**：`R_orth = cos²(b_A,b_B)`、`R_overlap = O_B = Σ|b_Ai b_Bi|/(‖b_A‖‖b_B‖)`，两者都在 [0,1]，λ∈{0.1,1} 在 family 间有可比量纲。

**drift 口径**：主指标 `D_B = ‖B_eff^final − B_eff^init‖_F / ‖B_eff^init‖_F`；`B_raw` 范数只作数值诊断。

smoke 证据（seed 11、o0、10 epoch、hybrid）：`O_B` 1.000→0.973（learnable）、→0.954（overlap0.1），B_eff 已在动、惩罚臂解耦更快。

## 2. Arms 与分阶段执行（不看结果调参）

### V21A（主实验，λ=0.1）

| arm | 初始化 | penalty |
|---|---|---|
| fixed-overlap | V20 a1 | —（复用 V20，1 seed 回归核验） |
| fixed-disjoint | V20 a0 | —（复用 V20，1 seed 回归核验） |
| learnable-overlap-init | overlap | none |
| learnable+orth | overlap | `λ·cos(B_A,B_B)²`，λ=0.1 |
| learnable+overlap | overlap | `λ·Σ_i|B_Ai B_Bi|`，λ=0.1 |

### V21B（无论 A 结果如何都执行）

`learnable+orth (λ=1)`、`learnable+overlap (λ=1)` 两个 robustness arms。

### V21C（初始化对照）

`learnable-overlap-init` 之外，增加 `learnable-random-init`（无正则），回答“最终解耦是否依赖从 overlap 出发的训练路径”。

### V21D（协议敏感性，不入主统计）

4 arms × 5 seeds × o0 × r∈{0,12.5%} = **40 流全部 Full BPTT**（`--b-grad full`）：fixed-overlap / fixed-disjoint / learnable / learnable+overlap(0.1)。只回答“所有参数都有长程信用时，V21 的解耦方向是否仍存在”。

### V21-trunc（负控，不入主统计）

5 seeds 一次性 diagnostic：验证 `max|∇B| < 1e-12`、`D_B = 0`，产出一张“TBPTT 切断 B 信用”的机制图，不跑全量流。

### 命名约定

- `orth` = **方向正交控制组**；V20 `overlap_a1` 已经 B_AᵀB_B≈0 但仍共享节点，因此正交不等于解耦。
- `overlap penalty` = 针对状态资源竞争的主正则（不要求单个 b_c 稀疏）。

## 3. 指标定义

### 3.1 B 侧（B_eff）

- 主指标 **幅度重叠**：`O_B = Σ_i |B_Ai B_Bi| / (‖B_A‖₂‖B_B‖₂)`（∈[0,1]，使用同一节点即高，符号无关）。
- 次指标：`|cos(B_A,B_B)|`。
- 可视化：`J_top16 = |Top16(B_A)∩Top16(B_B)| / |Top16(B_A)∪Top16(B_B)|`。
- 记录 `‖B_A−B_B‖`、每 stage 的 `O_B`，训练中**每 5 epoch** 轻量记录一次 `O_B`（分裂轨迹）。

### 3.2 状态侧（counterfactual 响应子空间）

同一批 sample/meta，构造：
- `Δh_A(t) = h_t(x_A,0) − h_t(0,0)`
- `Δh_B(t) = h_t(0,x_B) − h_t(0,0)`

对每个通道的响应矩阵做 top-r SVD（r=8），
`O_h(t) = (1/r)‖U_A^{(r)T} U_B^{(r)}‖_F²`（top-r principal-angle cosine² 均值，0=子空间正交，1=相同）。

记录 `t ∈ {2,4,8,20}`，**主判定 t=8**（输入注入完成且经 W/Route/Hold 传播）。

### 3.3 梯度侧

- 每个 stage transition 保存 `cos_B, cos_W, cos_Route, cos_Hold`（V20 口径只测后三者，V21 必须加 B）。
- 统一 scalar（保持与 V20 可比）：`C_∇ = −(cos_W + cos_Route + cos_Hold)/3`（越大越冲突）；`cos_B` 单独报告。
- 记录 `‖B^{stage i} − B^{stage i−1}‖`（B drift）。

### 3.4 任务侧

A'B'C'D'：matrix / acquisition / final_mean / forgetting；单任务门槛 ≥99%。

## 4. 预注册假设

**H1（V20 replication）**：`fixed-disjoint` 优于 `fixed-overlap`。
- 主条件 **r=12.5%**：Δforgetting ≤ −2pp（V20 实测约 −3.4pp）；
- r=0 为方向复核（V20 约 −1.8pp）。

**H2a（自发解耦）**：`learnable (λ=0)` 从与 `fixed-overlap` 完全相同的 B 初始化出发，任务梯度本身是否让 B 逃离 overlap：
- `O_B^final ≤ 0.75·O_B^init`，≥4/5 seeds（主判定）；
- 与 fixed-overlap 的 forgetting 差异作为功能面辅助。

**H2b（显式正则的增量）**：干净比较是 `learnable+penalty` vs `learnable (λ=0)`（两者同 hybrid、同 B credit、同初始化，只差 `R_B`）：
- 至少一个 penalty family 在 λ∈{0.1,1} 两档**方向一致**，且相对 λ=0：`ΔO_B ≤ −0.10` 或 `Δforgetting ≤ −2pp`（r=12.5%）之一，并报告 `Δcos_grad`；
- 相对 fixed-overlap 的改善作为次要报告（该比较同时改变了 fixed→learnable 与 B credit，不能单独归因于正则）。

**H3（机制链）**：跨 arm×seed（先对 3 orders 聚合，避免 pseudo-replication）：
- `ρ_s(O_B, O_h) ≥ 0.4`、`ρ_s(O_h, C_∇) ≥ 0.4`、`ρ_s(C_∇, F) ≥ 0.4`、`ρ_s(O_B, F) ≥ 0.4`（Spearman）；
- mediation 分析标记为 **exploratory**（5 seeds 不做硬门槛）。

## 5. 协议

A'B'C'D'，序列 20 步；5 seeds（11/22/33/44/55）；orders o0/o1/o2；r∈{0,12.5%}；65 轮/阶段、5 步截断、Adam 0.003、batch 128、buffer 32/任务；单任务门槛 ≥99%。训练与 CL 逻辑复用 v20 `experiment.py`，仅替换 B 的定义与加 penalty。

## 6. 执行顺序

1. **V21 protocol commit**：PLAN + 可运行代码（hybrid/full/trunc、penalty 归一、D_B）；不带正式结果。
2. **Phase 0 credit check（5 seeds）**：`trunc` 的 `|∇B|=0`、`hybrid` 的 `|∇B|>0`、`max|g_core^hybrid − g_core^V20| < 1e-6`（证明只改 B 的信用分配）。
3. **V21A 主实验**：fixed 两臂复用 V20 + checkpoint 回归；3 个 learnable arms 用 hybrid 全量（先单任务门槛，再 CL r=0/12.5%、3 orders、5 seeds）。
4. **V21B（λ=1）** 与 **V21C（random-init）** 按预注册执行。
5. **V21D（40 流 Full BPTT 敏感性）** 与 **V21-trunc（5 seeds 负控）**。
6. `analyze.py` 只做预注册判定：H1/H2a/H2b/H3 + 汇总表；不因某 λ 好而改称“主模型”。

## 7. 输出

- `results/{A,B,C}/*.json` + 检查点 + `metrics_*.json`；
- 机制轨迹：`O_B` 每 5 epoch、stage 级 `O_h/cos_*/B_drift`；
- 图：`b_overlap_trace.png`（O_B vs epoch/stage）、`state_overlap.png`（O_h vs t/arm）、`conflict_forgetting.png`（C_∇ vs F）、`decoupling_chain.png`（Spearman 链）。

## 8. 边界与风险

- 只改输入写入方式，不改 W/Route/Hold/Readout 结构；role2/3 恒 0。
- `O_h` 依赖 counterfactual 输入构造，meta 固定；r=8 与 t=8 为预注册主口径，其余为曲线。
- B 变成可训练参数后，B drift 与 cos_B 必须与旧指标同表报告，防止冲突“转移”未被看见。
- 5 seeds 下相关/中介只作机制证据，不作统计显著性结论。

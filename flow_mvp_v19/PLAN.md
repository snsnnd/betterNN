# 第十九轮：并行求解器可行性（Phase 0 诊断 → predictor 原型）

## 0. 背景

第十八轮用 **Gauss–Seidel 式 JVP 传播**（`block_affine.compose`：逐块循环，第 i 块 JVP 方向使用本遍刚算出的 h_i）证明：40 步外推的严格串行依赖可以用“预测器初值 + block-affine 重线性化”压到有效深度 12～15，精度 99.3～99.6%（Flow 99.82%）。

但有两个未计入/未证明的缺口，本轮先补齐再谈训练：

1. **V18 不是块并行求解器**。GS-JVP 的串行深度是 (K+1)·L，跨块方向依赖使它无法直接 scan；v17 的显式矩阵 scan 在 CPU 上比串行慢 143×（scan_check.json：0.098s vs 14.03s），且 predictor 成本未计入。
2. **真正的并行需要可组合的 affine operator** `(M_i, c_i)`，而 JVP 只能作用一个方向，不构成独立可组合的算子。

本轮训练一律后置：先回答“S + 低秩修正的结构化 operator 能否在精度、深度、GPU wall-clock、显存上成立”，再决定是否训练 predictor / joint training。

## 1. 已完成的 Phase 0 前置检查（seed 11、task 0、L=4、真轨迹）

Flow-v2 单步更新（`block_affine.block_map` 的 L=1 形式）：

```text
a_t      = 0.2·sigmoid(hold(meta_t))          # 与 h 无关
g_t      = sigmoid(Route([meta_t, pooled(h_t)]))   # pooled = h.sum(-1)/64，rank-4 通道
rec_t    = A(g_t) h_t,  A(g)[(d,j),(s,i)] = g[s,d]·W[s,i,d,j]
cand_t   = tanh(rec_t + x_t B)
h_{t+1}  = (1−a)h_t + a·cand_t
```

因此 `J_t = D_{1−a} + D_a D_{tanh'} [A(g_t) + U_t V_tᵀ]`，其中 Route 修正项经 `pooled(h)` 因式分解，**rank ≤ 4（精确）**。

实测（`/tmp/opencode/rank_check.py`，逐样本 `torch.func.jacrev`）：

| 命题 | 结果 |
|---|---|
| 单步 rank(J−S) ≤ 4 | 成立：前 4 个奇异值 O(1e-2~1e-1)，第 5 个起 1e-7 噪声底；t=0 时修正恒为 0（h=0） |
| 块 rank(J_block−S_block) ≤ 4L（L=4） | 成立：每块显著秩 6～9 ≪ 16，索引 16 后为噪声底 |
| scan tree 秩增长（B=10 两两组合） | 显著秩 13 → 9 → 16 → 16，**逐层饱和在 ~16**，未向 256 膨胀；r=16 截断相对误差 1e-5～1e-4，r=32 更低 |
| allowed 角色块支撑闭包 | `A²=A`（8/16 块），任意块 Jacobian 保持同支撑 |
| 单步 σ_max | 1.94 / 1.09 / 0.98 / 0.98 → 10.25 是 L=4 块乘积效应 |
| V18 缺失的 predictor 基线 | `eh(pre_states, true)` = 23.28（seed11/task0/4 样本）；E_h 近零分母失真，需配尺度不变指标 |

数值为 float32 autograd（噪声底 ~1e-7），仅单种子单任务，Phase 0B 需全量扫描确认。

## 2. Phase 0A：solver taxonomies

同架构、同参数、同测试集（V15 检查点 + v17 PreRoute 检查点），只比求解方式：

| 路线 | 块内 | 块间 | 真并行 | 作用 |
|---|---|---|---|---|
| A. GS-JVP | nonlinear 块函数 | 本遍新状态作为 JVP 方向 | ✗ | V18 已验证的精度上界 |
| B. structured operator | nonlinear | `S_i + U_i V_iᵀ` 仿射组合 + associative scan | ✓ | 目标实现 |
| C. fixed-center Jacobi | nonlinear | 固定 `p_i = predictor_i`，`h_{i+1}^{k+1} = F_i(p_i) + J_i(p_i)(h_i^k − p_i)` | ✓ 每遍 | 不构造矩阵的并行诊断 |
| D. dense M + scan | nonlinear | 显式 `J_i` + scan | ✓ | 成本上界参照（v17 已证昂贵） |

预注册说明：

- C 的线性部分是块对角移位算子，**幂零，最多 B 遍收敛**；其瓶颈是每遍只前进一个块（warm-start 版 block Jacobi），与 σ_max 无关。若改为中心每遍重线性化，属另一迭代，需单独标注。
- B 的精确形式不需要 SVD 截断：单步修正 rank ≤4 可解析累积；S 的乘积按 8/16 角色块稀疏计算。截断只用于 scan-tree 的再压缩。

## 3. Phase 0B：operator 结构测量

配置：5 seeds × 4 tasks × {20, 40} 步 × L ∈ {1, 2, 4, 5, 10}，每任务 256 样本（与 V15/V18 测试集一致）。

测：

1. `σ(J)`、`σ(J−λI)`、`σ(J−S)`；验证 `rank(J−S) ≤ 4L`（理论—数值一致性检查，报告显著秩定义与噪声底）。
2. scan tree 每层显著秩与截断再压缩：`r_max ∈ {8, 16, 32, 64}`，报告每层秩、截断误差、以及对最终 20/40 步状态/精度的影响。
3. 结构化 compose 成本单独记录（block matmul 次数与 FLOPs），不并入 apply 成本。

判定：

- 若 `J−S` 显著秩稳定 ≤ 4L 且 scan tree 秩饱和（不随层数增长）→ 路线 B 成立，进入 0C。
- 若秩随层数持续增长且截断使精度崩塌 → 结构化 operator 只适用于短块（L≤5、单/双块），路线 B 降级为“短块仿射 + 串行接续”。

## 4. Phase 0C：GPU 实现与计时协议

必须满足：

- block batch 化（`batch × block` flatten 后一次 `block_map`）；
- centers / JVP / Jacobian 全部 batch 化，scan 内无 Python 块循环；
- CUDA synchronize 后计时，预热，同 batch / dtype / device；
- 基线 = 同设备串行 Flow-v2（20/40 步）、PreRoute（scan 与串行两种实现）。

记录：latency、throughput、peak memory、FLOPs（含 compose 成本）、JVP/VJP 次数、operator bytes、有效深度。

规模：T ∈ {40, 128, 512} × batch ∈ {1, 8, 32}；N=256；若 0B 支持再补 N=512。

判定（0C，按优先级）：

1. accuracy（acc20/acc40）；
2. 达到 99.5% 所需 K；
3. GPU latency ≤ 串行 Flow（同 batch）；
4. peak memory 无数量级增长；
5. 报告 FLOPs，`c ≤ 2` 为优化目标而非生死线。

## 5. Phase 1：boundary predictor distillation（冻结 Flow）

- 输入 `Pψ(x, meta, t_i) → ĥ_i`，一次前向输出全部 B 个块边界，前馈、无递归（不得用 PreRoute 式 T 步递归）。
- 损失：相对 MSE `Σ_i ‖ĥ_i−h_i‖²/(‖h_i‖²+ε)`；报 `E_h` 与尺度不变的 `E_J = ‖J_i(ĥ_i−h_i)‖/(‖h_{i+1}‖+ε)`。
- 验收：`E_h`/`E_J` ≤ PreRoute 原始边界误差（本轮先补测该基线并稳健化），且不改变 Flow 任何参数。
- 无二阶梯度：梯度不过 solver。

## 6. Phase 2：冻结 solver 评估

predictor + solver（A/B/C），L ∈ {4,5}，K ∈ {1,2,3}；目标 `acc40 ≥ 99.5%`，给出 latency–accuracy Pareto。只有产生 wall-clock Pareto 优势（相对串行 Flow）才进入 Phase 3。

## 7. Phase 3：joint training（条件触发）

- 预注册两个版本：**full second-order**（保留 `∇θ(Jv)`）与 **stop-Jacobian**（`J → stopgrad(J)`，只让 loss 走零阶路径）。二者不得混为一个实验。
- 先测 `time(2nd-order)/time(1st-order)` 与 peak memory ratio；若二阶爆炸，不投入正式训练预算。
- 训练预算与推理预算分开报告（training FLOPs/wall-clock 与 inference FLOPs/latency 独立）。

## 8. 输出

- `operator_structure.json`（0B 谱与秩表）、`solver_bench.json`（0A/0C 精度-延迟-显存）、`predictor.json`（Phase 1）、图 `solver_pareto.png`；
- 报告 `REPORT.md` 自动生成，判定规则按本文件预注册。

## 9. 边界与风险

- 所有 Jacobian/谱结论目前基于 float32 autograd 与单种子；0B 全量扫描前不得当作定论。
- E_h 近零分母失真；统一采用尺度不变指标或剔除近零分母。
- 结构化 compose 的 FLOPs 可能显著高于串行 apply（块乘 ~4M MAC vs apply ~33k MAC），“深度↓”不等于“算力↓”。
- 不引入新门、不扩大 N、不改 Flow-v2 正式架构；所有新模块（predictor/solver）只作为外围执行机制。
- RNG/回归约定沿用仓库规范：增删模块需用 `_rng_pad` 保持初始化流一致；新实现与原路径逐位比对。

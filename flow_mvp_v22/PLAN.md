# 第二十二轮：Long-range Credit Attribution（到底哪个模块需要完整长程信用）

## 0. 背景与问题

V21 给出了一条非常干净的因果链证据：

- fixed-overlap + core 截断：r=12.5% 遗忘 9.04pp；
- fixed-overlap + **全 20 步 BPTT**：4.66pp（V21D）；
- learnable B + 全 20 步 BPTT：**1.83pp**（V21D），即解耦代价被消除；
- 但 learnable B + hybrid（只有 B 拿全 BPTT、core 仍 5 步截断）：**6.63pp**（V21A），几乎不降。

因此当前瓶颈不是"B 是否解耦"，而是**长程信用分配本身**：B 在改，但 W/Route/Hold 只看到最后 5 步的信用，无法与 B 形成跨时间的联合适应。V21 结束语把下一步明确为：

> 从 hybrid 的 6.63pp 到 Full 的 1.83pp，究竟给哪些组件补上完整时间信用后发生？

V22 不再改架构、不再动 B 的初始化/正则，只做一件事：**逐组打开/关闭 20 步信用窗口**，用 2³ 因子设计把"长程信用"归因到 `{W, Route, Hold}` 三个 core 模块上。B 恒为可训练且恒为 Full（V21 已证明 B 在截断下梯度恒为 0）。

## 1. 逐组信用窗口协议（V22 的唯一改动）

对每个训练 step、每个 batch，用**同一参数快照**做两条前向/反向：

1. **截断通道**：`detach_window=True`、`detach_period=5`（与 V20/V21 完全一致），`loss.backward()` 得到全部参数的 5 步信用 `g_trunc`；
2. **完整通道**：`m.eval()`、`detach_window=False`（20 步无损图），`torch.autograd.grad(loss_full, params_F)` 得到 Full 组参数的 20 步信用 `g_full`；
3. **按组拼装梯度**：参数组 ∈ Full 集 → 用 `g_full`；其余 → 用 `g_trunc`；
4. **双 optimizer、分别 clip**（与 V21 hybrid 完全相同的更新规则）：
   - `Opt_core = Adam(非 B 参数, lr=0.003)`，`clip_grad_norm_(core, 1)`；
   - `Opt_B = Adam(B_raw, lr=0.003)`，`clip_grad_norm_([B_raw], 1)`。

参数分组：`W` / `route`（`route.*`）/ `hold`（`hold.*`）/ `readout`（`headX/headY`，读出只在最后一步使用，两条通道梯度逐位相同，恒归 Full）/ `B`（`B_raw`，恒 Full）。

**arms（2³ 全因子，子集 S ⊆ {W, Route, Hold} 拿 Full）**：

| arm | Full 组 |
|---|---|
| `B` | B |
| `B+W` | B, W |
| `B+R` | B, Route |
| `B+H` | B, Hold |
| `B+W+R` | B, W, Route |
| `B+W+H` | B, W, Hold |
| `B+R+H` | B, Route, Hold |
| `All` | B, W, Route, Hold |

`B` 臂 = V21A learnable（hybrid）；`All` 臂 ≈ V21D learnable（40 流敏感性）。二者以新协议重跑，作核验锚点。

### 1.1 必须核验的回归（Phase 0）

- `B` 臂与 V21A `learnable`（种子 11–55、o0、r=12.5%）**逐位一致**：`max|Δ matrix| = 0`（两条前向/反向的调用顺序与 V21 hybrid 完全复刻）。
- `All` 臂与 V21D `learnable` 一致：`max|Δ matrix| ≤ 0.02`（V21D 用单 optimizer + 联合 clip，V22 用双 optimizer + 分别 clip；B 梯度范数 ~3e-3，clip 差异可忽略）。
- 逐组信用审计：至少一个 core 组的 `cos(g_5, g_20) < 0.9`，证明截断确实改变了信用方向（否则实验无意义）。

## 2. Phase 0：逐组 credit audit（不训练，回答"截断切掉了多少"）

对每个 checkpoint，在固定 batch 上分别取 `period ∈ {5, 10, 20}`（20 = 无 detach）：

- `cos_pg = cos(g_p(θ), g_20(θ))`、`ratio_pg = ‖g_p(θ)‖/‖g_20(θ)‖`，按组 `{W, route, hold, readout, B}` 记录。

checkpoint 来源：

- `init`：新初始化模型（种子 11–55），任务 0 的 batch；
- `single`：V21A `single_learnable_t{0..3}_{seed}.pt`，在各自任务上审计；
- `cl`：V21A `learnable_o0_r{0,12.5}_{seed}_stage{i}.pt`，在 stage i 的训练任务上审计。

输出 `results/credit_audit_*.json`，plot 出"各组的信用窗口曲线"。这一相位完全复用 V21 检查点，不新增训练。

## 3. 指标

- 任务侧：matrix / acquisition / final_mean / forgetting / BWT（定义与 V20/V21 相同）。
- B 侧：`O_B`、`D_B`（仅记录，不作本轮判定）。
- 训练中每 stage 记录一次 `credit_stats`（period 5/10/20 的分组 cos/ratio）与每 10 epoch 的 `b_trace`。
- 主口径：**r=12.5%、o0/o1/o2 三顺序按 seed 聚合**；r=0 只跑 o0 作负控。

## 4. 预注册假设（全部在 r=12.5%，除注明外）

记 `F(S)` 为 arm S 的 3 顺序聚合遗忘（pp）；对 seed 配对。

- `G = F(B) − F(All)`：hybrid → 全信用的总 gap。
- `closure(S) = (F(B) − F(S)) / G`：arm S 补上的比例。
- `ME_X = E[F(S): X∉S] − E[F(S): X∈S]`：给 X 打开 Full 的主效应。
- `LOO_X = F(All−X) − F(All)`（`All−W=B+R+H`、`All−R=B+W+H`、`All−H=B+W+R`）。

**H0（协议核验）**：`B` 逐位回归（max|Δ|=0）；`All` 对 V21D max|Δ matrix| ≤ 0.02；audit 存在 core 组 cos(g5,g20) < 0.9。

**H1（Route/Hold 是主要长程信用载体）**：`max(ME_R, ME_H) ≥ 1.5pp` 且 `max(ME_R, ME_H) ≥ ME_W + 1.0pp`，且该因子的 seed 配对差异 ≥8/10 同向。

**H2（W 单独不足）**：`closure(B+W) ≤ 0.25` 或 `ME_W ≤ 1.0pp`。

**H3（R+H 补上大半 gap）**：`closure(B+R+H) ≥ 0.5`。

**H4（必要性，leave-one-out）**：`max(LOO_R, LOO_H) ≥ 0.5·G`。

**H5（竞争假设：联合协同）**：若 `closure(B+R+H) < 0.5` 且所有二因子臂 `closure < 0.5`，则支持"没有单一子集（≤2 组）能补上 gap、必须 W+Route+Hold 联合协同"（此假设与 H1/H3 互斥，先跑先判）。

**H-neg（r=0 负控）**：所有 `|ME_X| ≤ 1.0pp`（无 replay 时信用窗口不改变遗忘）；若否，说明机制依赖 replay，需要重新解释。

所有阈值在 `analyze.py` 中硬编码；不因结果方向改称"主模型"。

## 5. 训练协议（除信用窗口外与 V21 完全一致）

- A'B'C'D'、序列 20 步、N=256、65 轮/阶段、Adam lr 0.003、batch 128、5 seeds 原协议 + 新增 seeds 66/77/88/99/110（共 10 seeds）。
- r ∈ {0, 12.5%}；r=12.5% 跑 o0/o1/o2，r=0 只跑 o0；每任务 buffer 32。
- learnable B 从 `_overlap_B(seed,1.0)` 初始化（所有 arm 同种子同初始化）；`B_eff` 归一化 `β=5.6`，写入域 role0∪role1。
- singles：5 个原种子 × 4 任务（表参考，不作主判定）。
- 单线程 `FLOW_THREADS=1`，按配置并行（28–30 进程）。

## 6. 执行顺序

1. 本 PLAN + `experiment.py`（含 `--credit-audit`）提交后（无正式结果）；
2. Phase 0：smoke（2 epoch）→ `B`/`All` 单流回归 → credit audit 全量；
3. 主运行：8 arms × 10 seeds × o0/o1/o2 × r{0,.125}；r=0 只 o0；singles 5 seeds；
4. `analyze.py`（H0–H5/H-neg + 汇总 + 自动 REPORT）→ `plot.py`；
5. 更新 `docs/EXPERIMENTS.md`、`README.md`、`docs/REPRODUCING.md`、`HANDOVER.md`。

## 7. 输出

- `results/*/*.json`（CL 行、single 行）+ `results/credit_audit_*.json` + `results/summary.json` + `verification.txt`；
- 图：`attribution.png`（各 arm 遗忘 + closure 条形）、`main_effects.png`（ME/LOO with seed dots）、`credit_audit.png`（逐组 cos/ratio 随 stage 曲线）；
- `REPORT.md` 自动生成。

## 8. 边界与风险

- 只改"哪些参数用哪条信用通道"，不改 forward 数值路径；readout 恒 Full（两条通道逐位相同）。
- 混合信用是每个参数独立的窗口选择，不是某种新优化器；B 恒 Full，避免与 V21 的解耦问题混淆。
- r=12.5% 的 o0 单顺序方差很大（V21 实测 seed 间 0–14pp），因此主判定必须三顺序聚合、10 seeds。
- 10 seeds 下主效应误差约 0.5–0.8pp；1.5–2pp 的阈值是"有意义但不苛刻"的水平，边界情形按方向一致性报告，不强行二分。
- 探索性：`F(S)` 的加性/超加性分解只作描述（8 个 arm 的完整 2³ 允许一阶交互的定性观察，不做正式 ANOVA）。

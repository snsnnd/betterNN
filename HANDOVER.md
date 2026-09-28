# 项目交接文档（HANDOVER）

给后续 agent / 研究者的交接说明。目标：一小时内理解项目是什么、做到哪一步、哪些结论可信、下一步从哪里接。

- 仓库：`git@github.com:snsnnd/betterNN.git`（main 分支，WSL 下 SSH 已配置）
- 最新轮次：**第 24 轮（含 V24B/V24C）**（Input Write Dynamics / Write Scheduler / Adaptive Temporal Compression，见 [flow_mvp_v24/REPORT.md](flow_mvp_v24/REPORT.md)）
- 当前正式架构：**Flow-v2 = W + Route + Hold + Readout**（66,162 可训练参数；B 固定）
- 当前最佳持续学习配方：**Flow-v2 + 每步 12.5% 样本 replay**（65 轮/阶段、A'B'C'D'）
- **并行/solver 线已冻结在 V19**；V20 B 拓扑；V21 可训练 B 解耦；V22 长程信用归因；V23 压力任务；V24 写入时间结构
- 信用分配结论（已定）：**core 用 5 步窗口足够**（V22；V23 无硬案例）；B 用 hybrid 双通道拿完整信用
- 写入结论（V24）：**单次瞬时写入是结构性缺陷**；等能量 FIR kernel（burst5/decay-slow）在长 T 与 Chain-select 上有大收益，且偏好 kernel 依任务而变。尝试的可学习标量 write gate（V24B）是**负结果**（未超最优固定 kernel、0/30 学到重要性）→ 下一步转向 **λ_t / 带预算约束的 scheduler**

---

## 1. 项目是什么

研究"一张循环底网 + 小型动态调制器"能否通过**控制信息写入、传播与保留**来完成任务，并聚焦持续学习（continual learning）问题。合成任务：序列里少量关键事件（t=0/1 写入数值 A/B，t=4 起给出规则 R），最终输出两个符号判断。

核心组件：

| 问题 | 组件 |
|---|---|
| 从哪里进入？ | `B`（输入投影；V21 研究了可训练版本，见 §5/§9） |
| 往哪里传播？ | `Route`（块/门控路由）+ `W`（可塑循环矩阵） |
| 保留多久？ | `Hold`（更新率 a∈(0,0.2)） |
| 从哪里读答案？ | `Readout`（只读目的角色 h[:,2] / h[:,3]） |

**Write 门已在第 14/15 轮被实验淘汰并从代码删除**（细节见 §8）。V21 把输入问题重新打开：B 可以学习，但 5 步截断下它拿不到任何梯度（§8）。

近期主线：V19 冻结并行/solver 线 → V20 发现“B 拓扑只改动力学不改任务表现” → V21 发现“B 解耦的收益取决于长程信用分配”。

---

## 2. 累计结论（可信 / 已否定）

### 已被实验支持

1. **少量 replay 几乎消除遗忘**：每步 12.5% 样本（16/128）即可把遗忘从 29pp 降到 ~1pp（v11）。
2. **Flow 的 replay 效率高于容量匹配的 GRU/RNN**：Flow 达遗忘 ≤3pp 需 ~6.7% replay；GRU 25% 仍未达到；原版 RNN 在该预算下未学会（v12）。
3. **旧任务策略分布在 Route+Hold+W，Readout 不承载策略**：换回旧 ctrl 恢复 A 75.74→91.39%，ctrl+W→99.75%，readout 无影响（v13/v14）。
4. **Write 门可删除**：随机固定/常数/真训练/局部监督五种方案无可测差异（v14）。
5. **梯度冲突是真实的**：r=0 时 Hold 的新旧任务梯度 cosine −0.63，replay 后 +0.30 等（v13）；但并非所有组件都转正。
6. **块仿射可压缩性（诊断）**：以 PreRoute 轨迹为并行预测器，2 次迭代重线性化、有效深度 12～15 可把 40 步外推恢复到 99.3～99.6%（Flow 99.82%）（v18）。
7. **可扫描性数学成立**：仿射块组合的顺序 vs Hillis–Steele 扫描误差 2.2e-9（v17）。
8. **块 Jacobian 有精确结构（v19）**：`J_t = S_t + C_t`，C 来自 `pooled(h)→Route`（rank ≤4）；块修正 `rank(J−S) ≤ 4L` × 5 seeds × 4 tasks 全量验证，**scan tree 秩逐层饱和在 ~12**，截断 r=16 相对误差 ~9e-6（近无损）。
9. **结构化 solver 与 V18 GS-JVP 数值等价（v19）**：同中心下 acc 完全一致；与 v18 存储 JSON（batch 256）max|Δacc|=0。
10. **输入拓扑只改动力学、不改任务表现（v20）**：等输入能量 `‖B_c‖₂=5.6` 下，single-random→distributed 的单任务 final 全 ≥0.994、r=0 遗忘均 ≈28–29pp（H1 Δ≤1pp）；但 `ρ_eff` 1.40→1.12、`G_max` 1.58→0.88、24 步衰减率 −0.029→+0.095（single-random 扰动净放大且 τ½ 截尾，distributed τ½≈6.6）。
11. **overlap 的代价集中在 W/Hold（v20）**：α=0.5/1 时 Δcos(W)=−0.22/−0.17、Δcos(Hold)=−0.11/−0.11；r=12.5% 下 overlap 族 final 0.94/0.91/0.92，低于主族 0.97–0.99。
12. **5 步截断下 B 的梯度恒为 0（v21）**：输入在 t=0/1 而 detach 在 t=5/10/15，`∇B≡0`（与 v13 Write 同源）；hybrid 协议（core 截断 / B 全 BPTT、双 optimizer、分别 clip）可在不改变 core 学习规则（梯度逐位相同）的前提下训练 B。
13. **B 自发解耦存在但不足以达标（v21）**：learnable (λ=0) 把 O_B 从 1.0 降到 init 的 0.83–0.91（0/5 ≤0.75）；overlap penalty 可强解耦（O_B→0.12/0.001）但 hybrid 下遗忘仅变 ±0.6pp。
14. **B 的完整信用是必要杠杆（v21；V22 修正）**：5 步截断下 `∇B≡0`，B 必须拿全 BPTT 才能学习；hybrid（core 截断 / B 全 BPTT）下 learnable B 把 fixed-overlap 的 3 顺序遗忘从 9.04pp 降到 5.4–6.6pp。V21D 的“Full-BPTT 把 overlap 代价从 4.66 压到 1.83pp”只在 o0 测得，**匹配口径下不可复现（见 16）**。
15. **长程信用归因是零结果（v22）**：把 `{W,Route,Hold}` 的 Full/T 信用做成 2³ 因子（B 恒 Full；10 seeds × 3 顺序）：G=F(B)−F(All)=**0.17pp**，主效应 ME_W/R/H=−0.14/+0.05/+0.36pp（同向 5/10、4/10、5/10）；没有单一子集（含 All）能稳定降低 r=12.5% 遗忘；r=0 负控所有 arm 24–25pp、|ME|≤0.4pp。
16. **V21 的 hybrid→Full gap 主要是聚合错配（v22）**：6.63pp（3 顺序 hybrid）vs 1.83pp（仅 o0 Full）= 表面 4.80pp；匹配同 5 seeds/3 顺序后 1.11pp、10 seeds/3 顺序 0.17pp；分顺序 B−All = +1.08（o0，5/10）、+1.39（o1，6/10）、−1.94（o2，4/10）。固定-overlap/解耦的结论（H1）不受影响（V20/V21 都是 3 顺序口径）。
17. **信用窗口损失真实存在但不构成遗忘瓶颈（v22）**：训练后 5 步梯度相对 20 步：W cos≈0.33（模长 6–10%）、hold≈0.50–0.65、route≈0.74–0.84；打开 Full 信用仍不改变遗忘——梯度几何失配不是功能重要性的可靠代理。
18. **主动构造压力任务仍未找到 core 长程信用硬案例（v23）**：Chain-select（瞬态脉冲 A/c1/distractor/B/c2、T∈{20,40,80,160}、条件选择输出）上，T≥40 时 K=5 的可达性 ≥ Full（T=40 5/5 vs 3/5；T=80 xor 4/5 vs 3/5；T=160 2/3 vs 2/3）。唯一干净的窗口效应是短 T 的脉冲-边界对齐（xor T=20：K5 可达 0/5、K10 5/5）。
19. **失败模式是"软选择平台/周期振荡"而非信用不足（v23）**：y0≈0.75–0.94，Full 同样不稳定（T=40/80 可达 3/5；T=160 出现 val 1.0→final 0.38 崩溃）；K=5 坏盆续训 Full 100 epochs 无法救回（0.750→0.733）。
20. **sustained 对照方向不一致（v23）**：把 c1/c2 持续放进 meta（命令在最后一步可见）后，Full 5/5 可达而 K5 只有 2/5（xor）/4/5（xorsw）；与 transient 方向相反，说明瓶颈不是"命令可见性"，而是选择计算的优化盆地。
21. **等能量写入时间结构是真实自由度（v24）**：`z_t=Σa_k B x_{t-k}`（`Σa²=1`，外部输入只出现一次）在固定 B 的延迟任务上收益随 T 增长：T=20/40 无差异、T=80 +3.5pp、**T=160 +5.3pp**（burst5，16/20 同向）；`single@T20` 与 V20 fixed-disjoint 逐位一致。
22. **单次瞬时写入是 Chain-select soft-selection 崩溃的主因（v24）**：V23 任务 T=80、core 仍 K=5、B 仍 hybrid 时，`single` 0.777（xor）/0.913（xorsw）→ `burst5` **0.971** / `decay-slow` **0.999**，single 的崩溃 seed 全部救回；`single` 与 V23 同 cell 逐位一致（10/10）。
23. **机制是末端可解码性而非扰动幅度（v24）**：||Δh|| retention 与 acc 不同向（全网格 r=0.08、T≥80 子集 r=0.42）；线性 probe 在 T/2 负相关（r=−0.47）、T−1 弱正相关（r=+0.46）。偏好 kernel 依任务而变（xor→burst5、xorsw→decay-slow），证明"写入时间策略"需要学习而不是固定。
24. **标量 write gate 是错误的第一自由度（v24B 负结果）**：`w_t=2σ(MLP(x_t,meta_t))`（初始 w≡1、不看 h、hybrid 全信用、core K=5）在 Chain-select T=80 上未能超过每任务最优固定 kernel（xor best sched 0.946 vs fixed burst5 0.971；xorsw 0.999=0.999）；H6 重要性 0/30（没有相对抑制 distractor；常见“全开放大”w_all≈1.45–1.56 或“只留末端事件”）；T=160 时 fixed burst5 自身也只有 0.777、scheduler 0.743，没有展示空间（V24A 的长 T 收益只在延迟任务上成立）。

### 已被否定/削弱

| 假设 | 结论 | 轮次 |
|---|---|---|
| 学习到的 Top-K 优于随机 Top-K | 否（−0.66±9.13pp，2/5 正） | v8 |
| 冻结 W 提供更好稳定性 | 否（配对数 1/15，最终低 3.5～4.9pp） | v11 |
| 网络越大越自然分区 | 否（支持集重叠训练后反而增加） | v7/v8 |
| Write 门学会"何时写入" | 否（5 步截断下梯度恒为 0，从未训练） | v13 |
| Route/Hold 策略锚点可替代样本 replay | 否（0.7～23KB 遗忘几乎不变；λ=50 也不改善） | v16 |
| 完全并行化不损失能力 | 部分（CL 接近，40 步外推 −8.8pp） | v17 |
| 两遍全局迭代可补回外推 | 否（更差：73.7 vs 90.4） | v17 |
| 结构化 affine scan 带 actual GPU 加速 | 否（eager 下慢 2~35×；跨块 Jacobian 乘积 fp32 溢出） | v19 |
| PreRoute 可作长序列 predictor | 否（T=128 max\|h\|=2e10、T=512 NaN） | v19 |
| multi-separate 降低无 replay 遗忘 | 否（multi8 −1.01pp、multi16 +0.02pp，门槛 −3pp） | v20 |
| distributed 提高可达性 | 否（随机输入 probe 全拓扑 r_eff≈1.1–1.2，0/5） | v20 |
| 初始谱尺度 s_W 影响训练后行为 | 否（0.7/0.9/1.1 任务与动力学均无差异） | v20 |
| B 任务梯度能自发强解耦（≤0.75·O_B^init） | 否（0.83–0.91，0/5） | v21 |
| hybrid 下显式解耦正则降低遗忘 | 否（Δforget ≤0.6pp，虽然 O_B→0.001） | v21 |
| orth penalty 有额外价值 | 否（init 已 |cos|≈0，ΔO_B≈−0.01） | v21 |
| Route/Hold 是长程信用的主要载体 | 否（ME_R=+0.05、ME_H=+0.36pp，G≈0；源自 v13 策略定位） | v22 |
| core 的 5 步窗口是遗忘瓶颈 | 否（2³ 因子匹配口径无效应，r=0 同样无） | v22 |
| V21D 的 o0 Full-BPTT 优势可外推到 3 顺序 | 否（o1 +1.39、o2 −1.94；10 seeds 聚合 0.17pp） | v22 |
| `cos(g5,g20)` 等梯度窗口失配可预测遗忘 | 否（W cos≈0.33 但 Full 无收益） | v22 |
| 长 T（到 160）下 5 步 core 窗口系统失败 | 否（T≥40 可达性 K5≥Full） | v23 |
| 指令持续可见（sustained）能消除 K5 失败 | 否（Full 5/5 vs K5 2/5，方向反了） | v23 |
| Full core credit 更稳定/更高可达 | 否（T=40/80 Full 可达仅 3/5，T=160 有崩溃） | v23 |
| K5 失败是缺长程信用 | 否（续训 Full 100ep 救不回，是吸引盆） | v23 |
| 单次瞬时写入不是结构问题 | 否（Chain-select K5：0.777→0.971 / 0.913→0.999） | v24 |
| ||Δh|| 扰动幅度可预测长程 acc | 否（全网格 r=0.08；T≥80 才 r=0.42） | v24 |
| 一种 kernel 通用最优 | 否（xor→burst5、xorsw→decay-slow） | v24 |
| 标量 write gate 能匹配/超过最优固定 kernel | 否（xor 0.946 vs 0.971；无一致优势） | v24B |
| write gate 学会相对抑制 distractor | 否（0/30；w 常>1 或只留 c2） | v24B |
| write gate 能在长 T 恢复 single 的崩溃 | 否（T160 0.743 ≈ single 0.770） | v24B |
| 自适应时间选择在同质延迟任务上也能发现固定 kernel 收益 | 否（α 停在 single：T80 0.734 vs fixed 0.898） | v24C |
| 延迟任务 T=80 上时间结构有效（预注册门槛） | 未过（+3.5pp < 5pp；T=160 才 +5.3pp） | v24 |

---

## 3. 当前正式架构 Flow-v2（精确规格）

代码基准：[flow_mvp_v15/experiment.py](flow_mvp_v15/experiment.py)（`FlowV2` 类）。

- `N=256`，4 个角色（来源 A/B、目的 X/Y），每角色 64 节点。
- `mask`：块结构 `[[1,0,1,1],[0,1,1,1],[0,0,1,0],[0,0,0,1]]`，块内/允许边再随机保留 30%；固定 buffer。
- `W`：256×256 可训练，初始化谱范数 0.9，训练中不约束。
- `B`：固定 buffer，`[2,256]`；通道 0 只写来源 A，通道 1 只写来源 B（正式架构中 B 固定；V21 的实验性可训练版本见 flow_mvp_v21，不改变本规格）。
- `hold`：Linear(7→4)，`a = 0.2·sigmoid(hold(meta_t))`（初始化输出 0.1）。
- `route`：MLP 11→16→16，输入 `q_t = [meta_t(7), pooled(h_t)(4)]`，输出 4×4 门 `g`。
- 更新：`cand = tanh(Σ_s (h⊙g) W + x_t B)`，`h ← (1−a)h + a·cand`。
- 读出：`headX` 只读 h[:,2]（目的 X），`headY` 只读 h[:,3]（目的 Y）。
- 训练：Adam lr 0.003、batch 128、每阶段 65 轮、5 步截断 BPTT（`t%5==0` 处 `h.detach()`）、固定最后 epoch、不做测试集选模。

**RNG 对齐约定**：v15 删除 Write 时用一个不注册的临时网络消耗相同的 RNG 流（`_rng_pad`），使同种子初始化与 v11/v14 完全一致；新方案若增删模块，务必用同样手法保持回归可比，否则必须先重跑基线。

---

## 4. Benchmark 与协议

### 任务集 A'B'C'D'（v10 起正式）

以 R=0/R=1 的输出通道表示：

| 任务 | R=0 | R=1 |
|---|---|---|
| A' | (A,B) | (B,A) |
| B' | (A,A) | (B,B) |
| C' | (A,B) | (A,A) |
| D' | (A,B) | (B,B) |

- 序列 20 步（外推测试 40 步）；`x` 仅 t=0/1 非零；`meta` 含任务 one-hot（t=0 可见）、归一化时间、R 标记与符号（t≥4 可见）。
- 单独可学性：scratch/adapt 全部 ~99.8%（v10）。**legacy A/B/C/D**（D 是 A 的规则逆映射）保留在 v8/v9，不要再混用。
- 训练/验证/测试：512/256/1024；验证集选预算：E90=48 轮、E95=65 轮，CL 固定 65 轮/阶段。
- 指标：matrix[stage][task]（两输出全对）、acquisition、final_mean、forgetting、BWT、epochs_to_90。

### Replay 协议（v11 起）

- 每步固定 128 样本：`(128−r·128) 当前 + r·128 replay`；每任务缓冲 32 样本；从所有旧任务样本均匀有放回采样。
- 当前任务的 batch 顺序用独立生成器，保证不同 replay 比例下当前数据一致。
- 常用比例：0 / 6.25%（8） / 12.5%（16） / 25%（32）。

---

## 5. 轮次索引（问题 → 结论 → 关键文件）

| 轮次 | 问题 | 关键结论 | 入口 |
|---|---|---|---|
| v1 | 冻结底网能否靠调制完成任务 | 调制器+读出 92.55%；严格控制器 55.87% | [flow_mvp](flow_mvp/) |
| v2 | 随机有效时刻与保持 | 长干扰 63.5→87.0（保持门）；通道关闭失败模式 | [flow_mvp_v2](flow_mvp_v2/) |
| v3 | 三门消融/W 训练/提示可靠性 | g111 87.5；无提示参考 70.5 | [flow_mvp_v3](flow_mvp_v3/) |
| v4 | Salience/路由/拓扑/延迟规则 | 动态路由任务有收益；两套分支不可混用 | [flow_mvp_v4](flow_mvp_v4/) |
| v5 | BPTT 窗口 × W 更新 | 5 步窗口 98.97%；冻结/慢速不足 | [flow_mvp_v5](flow_mvp_v5/) |
| v6 | 持续学习遗忘 | 最终 61±；独立学 C 99.0 vs 先学 B 后 62.2 | [flow_mvp_v6](flow_mvp_v6/) |
| v7 | 容量 × dense/topk | 512 topk 78.4；分区非学习涌现 | [flow_mvp_v7](flow_mvp_v7/) |
| v8 | Top-K 归因 | 固定≈学习，收益来自稀疏隔离 | [flow_mvp_v8](flow_mvp_v8/) |
| v9 | 遗忘定位 | replay 最有效；freeze_ctrl 崩溃 | [flow_mvp_v9](flow_mvp_v9/) |
| v10 | Benchmark 校准 | A'B'C'D' 单任务 99.8%；CL 基线 80.8/22.1 | [flow_mvp_v10](flow_mvp_v10/) |
| v11 | replay 比例 × 参数保护 | 6.25%→3.1pp 遗忘；freeze_W 无益 | [flow_mvp_v11](flow_mvp_v11/) |
| v12 | Flow vs GRU/RNN replay 效率 | Flow 6.7% 达 ≤3pp；GRU 25% 未达 | [flow_mvp_v12](flow_mvp_v12/) |
| v13 | replay 保护机制 | Write 从未训练；策略在 Route+Hold+W；缓冲阈值 | [flow_mvp_v13](flow_mvp_v13/) |
| v14 | Write 因果归因 | 五种 Write 方案无可测差异 → 删除 | [flow_mvp_v14](flow_mvp_v14/) |
| v15 | Flow-v2 基线 | 77.48/97.03/99.44，冻结架构 | [flow_mvp_v15](flow_mvp_v15/) |
| v16 | 策略锚点 vs 样本 replay | 策略锚点无用（0.7～23KB 平线）；11.8KB 样本达 ≤3pp | [flow_mvp_v16](flow_mvp_v16/) |
| v17 | 可并行动力学 | 预计算 Route CL 接近、外推 −8.8pp；tanh 承重 | [flow_mvp_v17](flow_mvp_v17/) |
| v18 | 块仿射可压缩性（诊断） | 预测器+2 次重线性化深度 12～15 恢复外推 | [flow_mvp_v18](flow_mvp_v18/) |
| v19 | 并行求解器可行性 | 结构：rank≤4/块≤4L、scan tree 秩饱和~12、r=16 无损；0C：eager 无加速、scan 溢出、PreRoute 长 T 发散 → 并行线冻结 | [flow_mvp_v19](flow_mvp_v19/) |
| v20 | B 输入拓扑 × 动力学 | H1 否（任务侧等能量无关）；动力学强效应（ρ_eff/G_max/lifetime）；overlap 增 W/Hold 冲突、replay 下变差；s_W 零结果 | [flow_mvp_v20](flow_mvp_v20/) |
| v21 | Adaptive Input Decoupling | 5 步截断下 ∇B≡0；hybrid（core 截断/B 全 BPTT）core 梯度与 V20 逐位一致；H2a 否、overlap penalty 强解耦但 hybrid 下不降遗忘；Full-BPTT 敏感性中解耦消除 overlap 代价（4.66→1.83pp） | [flow_mvp_v21](flow_mvp_v21/) |
| v22 | Long-range Credit Attribution | `{W,Route,Hold}` 的 2³ Full/T 信用因子：匹配口径下归因零结果（G=0.17pp、|ME|≤0.36pp）；V21 的 hybrid→Full gap 是 3 顺序 vs o0-only 的聚合错配（匹配后 1.11pp/5 seeds、0.17pp/10 seeds）；r=0 无效应；credit audit 显示窗口损失真实但不预测遗忘 | [flow_mvp_v22](flow_mvp_v22/) |
| v23 | Credit-Stress Benchmark | Chain-select 压力任务（瞬态脉冲 + T 到 160）主动找 core 窗口硬案例：未找到（T≥40 K5 可达性 ≥ Full）；唯一窗口效应是短 T 脉冲对齐；主失败是"软选择平台/振荡"且 Full 也中招、坏盆救不回（0.750→0.733） | [flow_mvp_v23](flow_mvp_v23/) |
| v24 | Input Write Dynamics | 等能量 FIR write kernel：长 T 收益随 T 增长（T160 +5.3pp）；Chain-select single 0.777/0.913 → burst5 0.971 / decay-slow 0.999。V24B 标量 write gate 负；V24C kernel bank selector 在 Chain-select 上成立（0.998/1.000，自动分任务策略），延迟任务上停在 single（HC3 未过） | [flow_mvp_v24](flow_mvp_v24/) |

大部分轮次的详细数字、图表和边界在各自的 `REPORT.md` 与根 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。

---

## 6. 仓库结构

```text
betterNN/
├── README.md                 # 项目总览与入口（先读）
├── HANDOVER.md               # 本文件
├── docs/
│   ├── EXPERIMENTS.md        # 21 轮索引 + 关键结果表
│   ├── ARCHITECTURE.md       # 模型/代码结构/指标定义
│   ├── FRAMEWORK_DESIGN.md   # 设计原理、梯度路径、改进方向
│   └── REPRODUCING.md        # 环境与逐轮命令
├── flow_mvp/ … flow_mvp_v24/ # 每轮独立目录（代码+结果+报告）
├── historical_deliveries/    # 第1～6轮历史 ZIP
├── .venv/                    # 运行环境（未入库；见 §7）
└── SHA256SUMS.txt / MANIFEST.json / COUNTS.json  # 原归档校验（只覆盖第1～7轮）
```

每轮目录约定：`PLAN.md`（预注册设计）、`experiment.py`（训练/数据）、`analyze.py`（全量重载核验 + 汇总 + 自动写 `REPORT.md`/`verification.txt`）、`plot.py`、`README.md`、`requirements.txt`、`results/`（JSON + `.pt`）。

---

## 7. 环境与运行

### 环境

- WSL Linux；项目在 `/mnt/d/betterNN`（9p → NTFS，注意小文件写入慢）。
- `.venv`：uv 创建，Python 3.11.15，**torch 2.14.0+cu130**、numpy 2.4.6、matplotlib 3.11.2。
  安装：`uv venv .venv --python 3.11 && uv pip install --python .venv/bin/python torch==2.14.0 numpy matplotlib`
- GPU：RTX 4060 Laptop（8G，WSL CUDA 可用）。**N=256 时 CPU 更快**（内核启动开销占主导）；N≥512 或需要大规模时才用 GPU。
- CPU：32 核。轻量实验每进程单线程（`FLOW_THREADS=1`），按**配置并行**（常规 8～30 进程）。
- 确定性：所有脚本 `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`；row 里记录 `threads/device`。线程数不同不保证逐位一致。

### 典型运行模式

- 单轮完整复现：见 [docs/REPRODUCING.md](docs/REPRODUCING.md)。
- 部分运行用 `--out smoke`（会写自己的 summary/report，不覆盖主结果）。
- 大批量扫描按方法/种子拆成多个后台进程：用 `setsid ... > logs/X.log 2>&1 < /dev/null &` 启动（避免用 `pkill -f` 清进程——模式会匹配到当前 shell 自身）；完成后跑 `analyze.py`。
- 重负载参考：v11 150 流/6 进程约 1 小时；v20 = 700 流、26 进程约 1 小时；v21 = 90 learnable hybrid 流（每步 2 次前向/反向）、30 进程约 50 分钟；v22 = 320 条 r=12.5% 流 + 80 条 r=0 + 160 单任务，28 进程约 33 分钟（内存吃紧，`--jobs 10` 更稳）；v23 = 149 条单任务流（T 到 160），T=160 单流 20–45 min、RSS≈1.1GB（并发 ≤4，其余 T 用 6 进程）；v24 = 430 条流（Phase A 400 条约 20 min/8 进程；Phase B 30 条 K=5×500ep 约 33 min/6 进程）。

---

## 8. 已知坑与经验（务必先读）

1. **t=0/1 注入的模块在 5 步截断下梯度恒为 0**：v7～v13 的 Write 从未被训练（v13 发现）；v21 实测 B 同样 `∇B≡0`（5 seeds），因为 detach 在 t=5/10/15。阅读旧报告时把 Write 当作“随机固定调制”；要让 B/Write 学习必须给它们全 BPTT 信用（v21 的 hybrid：core 截断 / B 全 BPTT，双 optimizer、分别 clip）。
2. **v4 有两套研究分支**（routing.py/salience.py 与 experiments.py 系列），指标格式不同，不能混成一个排行榜。
3. **RNG 对齐**：增删模块会改变后续模块初始化；用 `_rng_pad` 手法保持流一致，否则与旧结果不可逐位比较。
4. **新代码的强制回归**：任何声称等价的路径都要与旧基线逐位比对（如 v17 的 flowv2 对 v15、v16 的 none 对 v15、v18 的 oracle 恒等）。
5. **块 Jacobi 的教训**：多遍状态传播每遍只前进一个块，K<B 时末块入口恒零（见 v18/block_jacobi.py）。不要用这种"并行"实现。
6. **文件命名冲突**：同一目录下不同 λ/配置若共用文件名会互相覆盖（v16 的 λ 稳健性检查踩过）；用独立 `--out` 目录。
7. **`analyze.py` 要求完整结果集**：缺文件会断言失败；部分结果只能用对应过滤参数或独立目录。
8. **analyze 会重写 REPORT**：自动生成的报告不包含你手写的后补段落；要保留就在工作副本里跑。
9. **9p 磁盘慢**：每阶段写检查点会显著拖慢墙钟；能少存就少存，或写到 `~/`（ext4）再归档。
10. **v7 diagnostics 的 `h[:,4:]` 是时间维**（trace 带时间轴），不是角色维；读旧代码注意。
11. **`torch.func.jvp`/`vjp` 需要 eval 模式且函数纯**（v18 用法可直接参考）。
12. **推送**：仓库含大量 `.pt`（现约 5GB+），v20/v21 单次 push 约 600MB～1GB；用长超时（30 分钟），不要中途打断（中断后重推即可）。
13. **可训练模块 + 参数级诊断**：v21 中 learnable B 的 `eff_B` 用 cat 拼接而非原地赋值（原地写入不进计算图）；A/B 通道指标用 `O_B`（幅度重叠，符号无关）而不是支撑交集（dense 参数下无信息）；drift 用 `B_eff` 相对量，raw 范数只作诊断。
14. **后台进程管理**：启动用 `setsid ... < /dev/null &`；清理时按 PID（`pgrep` + `kill`），不要 `pkill -f 'results/B'` 这类会匹配到当前 shell 命令自身的模式。
15. **聚合口径必须匹配（V22 教训）**：V21 的“6.63→1.83pp”是 3 顺序 vs o0-only 的错配；V22 匹配后 gap 只剩 1.11pp（5 seeds）/0.17pp（10 seeds）。比较信用/协议时必须同 seeds、同 orders、同聚合方式；单顺序（尤其 o0）方差极大（0–17pp），不能单独下结论。
16. **逐组信用窗口实现**：V22 在同一参数快照跑两条通道（period=5 + 20 步），按组拼装梯度；`B` 臂与 V21 hybrid 逐位一致（`verify_protocol.py` 固化）；`All` 与 V21D 矩阵差 0.038（V21D 联合 clip vs V22 分别 clip；实测 clip 会生效 max pre-clip norm≈1.9–2.5），但遗忘均值一致（1.95 vs 1.83）。
17. **内存/并行**：本机 5GB，`--jobs 28` 在 9p 缓存+子进程下会 thrash（约 230s/流）；`--jobs 10` 稳定（约 100s/流）。重负载前先 `free -g`。
18. **r=0 别带全 orders**：`--orders o0 o1 o2 --ratios 0 .125` 会把 r=0 跑满 3 顺序（预注册只要 o0，多一倍开销）；负控用 `--orders o0 --ratios 0`。
19. **固定最后 epoch 在震荡任务上会误判（V23 教训）**：Chain-select 存在"软选择平台/周期振荡"，部分流 val 0.99 → final 0.5；报表必须同时给 `best-val reach`（只用验证集，非测试集选模）以区分**可达性**与**稳定性**，否则会把优化问题读成信用问题。
20. **T=160 资源与并发（V23）**：单流 500 epochs ≈20–45 min、RSS≈1.1GB；并发 ≤4，否则 9p 缓存 + 子进程会 swap。T≤80 单流 RSS≈0.3–0.6GB，可 6 进程。
21. **等能量控制（V24）**：write kernel 必须 `Σa²=1`，否则 burst 变好可能只是灌了更多能量；`single` 要跳过卷积以保持与旧轮的逐位回归（V20 fixed-disjoint、V23 K5 两个锚点都查）。
22. **幅度指标会骗人（V24）**：`||Δh_t||` 的 retention 与 acc 不同向（系统会放大扰动，single 的振幅反而更大更高）；用**线性 probe 的可解码性**（前 512 拟合/后 512 评估）作为“信息还在不在”的指标。
23. **符号任务里“写多强”有规范自由度（V24B）**：`y=sign(v)` 对输入做正标量缩放不改变可解性，内部增益可以补偿；因此标量 write gate 的 loss 平台很大、梯度不指向“相对重要性”，实测 0/30 学会抑制 distractor，反而常把 w 推到 >1 或整体压缩。设计 write scheduler 时要么给写入**预算/竞争约束**，要么改做 trace shaping（λ_t），否则不要期望它自动学到 importance。
24. **scheduler 协议核验（V24B/C）**：`verify_sched.py`（w_t 版）检查 ① 加 controller 不改变已有参数（maxdiff=0）、② 初始 w≡1 且 forward 逐值一致、③ 截断 ∇w=0/完整 ∇w≠0；`verify_bank.py`（bank 版）检查 base 等价、初始 α≈[0.93,0.017…]、forward≈fixed single、单脉冲总写入能量 rel diff≈1e-7、∇α 非零。任何新 scheduler 先过这三类检查。
25. **不要在大张量上做 in-place 切片赋值（V24C 教训）**：`out[:, s+tau] += ...` 会让 autograd 为每个 op 做整张量 `fill_`/`copy_`，backward 慢 8–10×（实测 bank T=80：0.79s/step → 函数式 `F.pad`+sum 0.15s/step）。构造时间移位用 `F.pad`/函数式切片，热路径不要 indexed in-place。
26. **自适应时间压缩在 Chain-select 上成立（v24C）**：kernel bank selector（α=softmax(MLP(x,meta))、逐事件 Σa²=1、无 w_t、core K=5）同一配置自动学到 xor“全事件铺开”（0.998，超 fixed burst5 0.971）与 xorsw“只给 c2 长 kernel”（1.000），HC1/HC2 通过。
27. **同质延迟任务上 selector 不移动（v24C）**：delay T80/T160 上 α 停在 single（初始化 bias 强局部最优），T80 task0 0.734 vs best fixed 0.898、T160 0.572 vs 0.606（HC3 未过）——收益需要 core 协同适应，65 epochs 内没被发现；下一步 V24D 加探索/退火或连续 λ。

---

## 9. 未决问题与下一步候选

按优先级：

1. **V24 已完成（Input Write Dynamics）**：等能量 FIR write kernel 在长 T 延迟任务上收益随 T 增长（T160 +5.3pp），在 V23 Chain-select（core 仍 K=5、B 仍 hybrid）上把 `single` 0.777/0.913 提到 `burst5` 0.971 / `decay-slow` 0.999，且偏好 kernel 依任务而变。**结论：单次瞬时写入是结构性缺陷；写入时间策略需要学习。** 下一步：
   - **V24B 已完成（标量 write gate，负结果）**：`w_t=2σ(MLP(x_t,meta_t))`（不看 h、初始 w≡1、hybrid 全信用）既没有超过每任务最优固定 kernel（xor 0.946 vs 0.971），也没有一条流学会相对抑制 distractor（0/30）。机制：符号任务对正标量缩放不敏感（内部增益可补偿），加法写入没有“写重了挤占别的信息”的竞争 → 相对重要性没有梯度。**结论：标量 input gate 是错误的自由度，不要再调它的容量/结构。**
   - **V24C 已完成（Adaptive Temporal Compression, kernel bank selector，结果 A + 例外）**：`α=softmax(MLP(x,meta))` 在 5 kernel 上选择、逐事件等能量、无 w_t；Chain T=80 上 xor 0.998（超 fixed burst5 0.971）、xorsw 1.000，自动学到“全铺开 vs 只给 c2 长 kernel”两种策略（HC1/HC2 通过）；但同质延迟任务上 α 停在 single（HC3 未过）。
   - **V24D（下一步）：连续 `λ_t` + 探索/退火**：`a_{t,τ}=c(λ_t)λ_t^τ` 等能量归一化；重点解决“初始化 single 是强局部最优、selector 不移动”的问题（延迟任务 +16pp 留给固定 kernel）；先只回答“连续 λ 是否比离散 bank 更能发现收益”。
   - **三个自由度按序验证**：Where（已答）→ When/How much（V24A 固定 kernel 有效 / V24B 标量 gate 无效）→ How long（V24C）；最后才是 Flow-v3（动态写哪里 `α_t` + 可并行 substrate）。
   - 机制补充：末端可解码性（probe T−1）比扰动幅度更能解释收益；新方法都应用 probe 而非 ||Δh|| 做机制指标。
2. **并行/solver 线**：冻结在 V19。若要重开，应改走 coarse+fine（parareal/多重网格）而不是 exact affine scan；先补 optimized serial 基线（torch.compile/CUDA Graph）再谈对比。
3. **Storage efficiency（Flow vs GRU）**：v12 只测了 replay band 效率，缓冲大小/存储效率尚未与 GRU 对比。
4. **State replay**：v16 测了样本与策略锚点，状态 h 的蒸馏未测。
5. 更远期：结构生长/修剪、频率/脉冲/相位调制（都要求先固定平均参数或信息预算）。

当前不建议：继续加新门、扩大规模（N≥512）、为 core 加长程信用机制，或回到"固定 kernel 再扫参数"（V24 已证明偏好依任务，继续扫固定值收益有限）；把 B 的可训练版直接塞回纯 5 步截断协议仍不可行（∇B≡0，必须走 hybrid 或等价双通道）。

---

## 10. 新 agent 第一小时清单

1. 读本文件 + [README.md](README.md) + [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。
2. 读七份最关键的实测报告：[v15](flow_mvp_v15/REPORT.md)（架构基线）、[v16](flow_mvp_v16/REPORT.md)（策略记忆负结果）、[v20](flow_mvp_v20/REPORT.md)（B 拓扑 × 动力学）、[v21](flow_mvp_v21/REPORT.md)（可训练 B 与 hybrid 信用分配）、[v22](flow_mvp_v22/REPORT.md)（长程信用归因）、[v23](flow_mvp_v23/REPORT.md)（Credit-Stress Benchmark）、[v24](flow_mvp_v24/REPORT.md)（Input Write Dynamics 与 write kernel）。
3. 检查环境与一致性：
   ```bash
   git status -sb && git ls-remote origin | head -2
   sha256sum --quiet -c SHA256SUMS.txt
   .venv/bin/python -c "import torch; print(torch.__version__)"
   cd flow_mvp_v22 && /mnt/d/betterNN/.venv/bin/python analyze.py | tail -3    # V22 缓存指标，约 5 秒
   cd flow_mvp_v22 && /mnt/d/betterNN/.venv/bin/python verify_protocol.py     # B 逐位回归 + 梯度等价 + clip 生效
   cd flow_mvp_v23 && /mnt/d/betterNN/.venv/bin/python analyze.py | tail -8    # V23 假设判定（读缓存 JSON）
   cd flow_mvp_v24 && /mnt/d/betterNN/.venv/bin/python analyze.py | tail -10   # V24 假设判定（H1–H7 + HC1–HC3，读缓存 JSON）
   ```
4. 复核当前最佳配方（可选）：v15 smoke `--epochs 2 --seeds 11 --out smoke` 或 v11 `--ratios 0 .125` 小规模。
5. 从 §9 选一个方向，按每轮约定新建 `flow_mvp_v25/`，先写 `PLAN.md`（含预注册判定规则），再写代码；完成后更新 `docs/EXPERIMENTS.md`/`README.md`/`REPRODUCING.md` 并 push。

---

## 11. 文档维护规则

- 每轮结束：更新 `docs/EXPERIMENTS.md`（加一节）、根 `README.md`（结论 + 目录 + 链接）、`docs/REPRODUCING.md`（运行命令）。
- 本地链接检查（提交前跑一次）：
  ```bash
  python3 - <<'PY'
  from pathlib import Path; import re
  root=Path('/mnt/d/betterNN')
  files=[root/'README.md',*sorted((root/'docs').glob('*.md'))]
  for p in files:
      for t in re.findall(r'\]\(([^)]+)\)',p.read_text()):
          if '://' not in t and not t.startswith('#'):
              assert (p.parent/t.split('#')[0]).exists(),(p,t)
  print('links OK')
  PY
  ```
- 提交信息包含轮次与关键发现，例如：`第二十二轮：<问题>（<关键数字>）`。
- HANDOVER 每轮同步：头部轮次/commit、§2 结论与否定表、§5 索引、§6 目录、§8 坑、§9 下一步、§10 清单。

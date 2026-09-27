# 项目交接文档（HANDOVER）

给后续 agent / 研究者的交接说明。目标：一小时内理解项目是什么、做到哪一步、哪些结论可信、下一步从哪里接。

- 仓库：`git@github.com:snsnnd/betterNN.git`（main 分支，WSL 下 SSH 已配置）
- 最新轮次：**第 20 轮**（B 输入拓扑 × 动力学，见 [flow_mvp_v20/REPORT.md](flow_mvp_v20/REPORT.md)）
- 当前正式架构：**Flow-v2 = W + Route + Hold + Readout**（66,162 可训练参数）
- 当前最佳持续学习配方：**Flow-v2 + 每步 12.5% 样本 replay**（65 轮/阶段、A'B'C'D'）
- **并行/solver 线已冻结在 V19**（结构结论保留，GPU 加速负结果）；V20 完成 B 拓扑 × 动力学主扫描

---

## 1. 项目是什么

研究"一张循环底网 + 小型动态调制器"能否通过**控制信息写入、传播与保留**来完成任务，并聚焦持续学习（continual learning）问题。合成任务：序列里少量关键事件（t=0/1 写入数值 A/B，t=4 起给出规则 R），最终输出两个符号判断。

核心组件：

| 问题 | 组件 |
|---|---|
| 从哪里进入？ | `B`（固定输入投影） |
| 往哪里传播？ | `Route`（块/门控路由）+ `W`（可塑循环矩阵） |
| 保留多久？ | `Hold`（更新率 a∈(0,0.2)） |
| 从哪里读答案？ | `Readout`（只读目的角色 h[:,2] / h[:,3]） |

**Write 门已在第 14/15 轮被实验淘汰并从代码删除**（细节见 §8）。

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

---

## 3. 当前正式架构 Flow-v2（精确规格）

代码基准：[flow_mvp_v15/experiment.py](flow_mvp_v15/experiment.py)（`FlowV2` 类）。

- `N=256`，4 个角色（来源 A/B、目的 X/Y），每角色 64 节点。
- `mask`：块结构 `[[1,0,1,1],[0,1,1,1],[0,0,1,0],[0,0,0,1]]`，块内/允许边再随机保留 30%；固定 buffer。
- `W`：256×256 可训练，初始化谱范数 0.9，训练中不约束。
- `B`：固定 buffer，`[2,256]`；通道 0 只写来源 A，通道 1 只写来源 B。
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

大部分轮次的详细数字、图表和边界在各自的 `REPORT.md` 与根 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。

---

## 6. 仓库结构

```text
betterNN/
├── README.md                 # 项目总览与入口（先读）
├── HANDOVER.md               # 本文件
├── docs/
│   ├── EXPERIMENTS.md        # 18 轮索引 + 关键结果表
│   ├── ARCHITECTURE.md       # 模型/代码结构/指标定义
│   ├── FRAMEWORK_DESIGN.md   # 设计原理、梯度路径、改进方向
│   └── REPRODUCING.md        # 环境与逐轮命令
├── flow_mvp/ … flow_mvp_v18/ # 每轮独立目录（代码+结果+报告）
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
- CPU：32 核。轻量实验每进程单线程（`FLOW_THREADS=1`），按**配置并行**（3～6 进程）已足够快。
- 确定性：所有脚本 `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`；row 里记录 `threads/device`。线程数不同不保证逐位一致。

### 典型运行模式

- 单轮完整复现：见 [docs/REPRODUCING.md](docs/REPRODUCING.md)。
- 部分运行用 `--out smoke`（会写自己的 summary/report，不覆盖主结果）。
- 大批量扫描（如 v16 的 330 条流）按方法/种子拆成多个后台进程，最后 `wait`，再跑 `analyze.py`。
- 重负载参考：v16 = 1320 个阶段、6 进程约 1 小时；v17 因 iter 两遍 rollout 约 50 分钟。

---

## 8. 已知坑与经验（务必先读）

1. **Write 已删除，但历史解释仍受影响**：v7～v13 的 Write 从未被训练（5 步截断切断了 t=0/1 的梯度路径）。阅读旧报告时把 Write 当作"随机固定调制"。
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
12. **推送**：仓库含大量 `.pt`（现约 2.7GB），一次 push 可能 30 秒～3 分钟；用长超时，不要中途打断（中断后重推即可）。

---

## 9. 未决问题与下一步候选

按优先级：

1. **V20 已完成（B 输入拓扑 × 动力学）**：任务侧等能量无关（H1 否、s_W 零结果），动力学侧强效应（单点输入放大扰动、分布式快速衰减）；overlap 增 W/Hold 梯度冲突并在 replay 下变差。下一步候选：
   - **可达性度量需要重做**：当前随机输入 probe 全部退化为 `r_eff≈1.1–1.2`（被主导方向支配）；可改输入—状态增益谱、闭环频率响应或多点脉冲探针后再谈 H2。
   - **任务侧瓶颈不在输入入口**：等能量拓扑不改变单任务/CL，说明要体现动力学差异需要改任务协议（更长序列、多事件写入、部分可观测）或转向记忆设计（多时间尺度、快慢状态）。
2. **并行/solver 线**：冻结在 V19。若要重开，应改走 coarse+fine（parareal/多重网格）而不是 exact affine scan；先补 optimized serial 基线（torch.compile/CUDA Graph）再谈对比。
4. **Storage efficiency（Flow vs GRU）**：v12 只测了 replay band 效率，缓冲大小/存储效率尚未与 GRU 对比。
5. **State replay**：v16 测了样本与策略锚点，状态 h 的蒸馏未测。
6. 更远期：结构生长/修剪、频率/脉冲/相位调制（都要求先固定平均参数或信息预算）。

当前不建议：继续加新门、扩大规模（N≥512）或在未解决"为什么 replay 这么有效"之前做复杂 consolidation。

---

## 10. 新 agent 第一小时清单

1. 读本文件 + [README.md](README.md) + [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。
2. 读三份最关键的实测报告：[v15](flow_mvp_v15/REPORT.md)（架构基线）、[v16](flow_mvp_v16/REPORT.md)（策略记忆负结果）、[v20](flow_mvp_v20/REPORT.md)（B 拓扑 × 动力学）。
3. 检查环境与一致性：
   ```bash
   git status -sb && git ls-remote origin | head -2
   sha256sum --quiet -c SHA256SUMS.txt
   .venv/bin/python -c "import torch; print(torch.__version__)"
   cd flow_mvp_v20 && /mnt/d/betterNN/.venv/bin/python analyze.py | tail -3
   ```
   （v20 analyze 依赖已缓存指标，约 15 秒，可用作环境冒烟。）
4. 复核当前最佳配方（可选）：v15 smoke `--epochs 2 --seeds 11 --out smoke` 或 v11 `--ratios 0 .125` 小规模。
5. 从 §9 选一个方向，按每轮约定新建 `flow_mvp_v19/`，先写 `PLAN.md`（含判定规则），再写代码；完成后更新 `docs/EXPERIMENTS.md`/`README.md` 并 push。

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
- 提交信息包含轮次与关键发现，例如：`第十九轮：<问题>（<关键数字>）`。

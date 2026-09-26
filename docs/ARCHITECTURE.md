# 架构与代码说明

[返回项目首页](../README.md)

若希望先理解“为什么这样设计、门控如何影响底层、梯度如何传递”，请阅读[框架设计原理与改进方案](FRAMEWORK_DESIGN.md)。本文侧重现有代码结构和接口。

## 1. 工程组织

项目采用“每轮独立目录”的研究代码结构。数据生成、模型定义和训练逻辑通常集中在一个入口脚本中，分析和绘图另设脚本。实验目录不是统一 Python 包，许多脚本使用本目录的裸模块导入和相对路径。

技术栈：

- **PyTorch**：张量运算、循环模型、自动微分、Adam 和检查点。
- **NumPy**：指标聚合及数值分析。
- **Matplotlib**：学习曲线、对照图和热图。
- **SciPy / NetworkX**：主要用于第四轮统计和图结构分析。
- **JSON / CSV / NPZ / PT**：配置、指标、轨迹及模型存储。

早期代码副本承担版本隔离作用：第三轮的 `v2_reference.py` 是第二轮快照；第五、六轮的 `base_routing.py` 来自第四轮局部模块实现；第六轮还保留 `window_models.py`。修改某轮实现不会自动同步到其他轮次。

## 2. 核心概念

| 概念 | 含义 | 实际作用 |
|---|---|---|
| 状态 `h` | 循环网络节点的连续数值 | 在时间步之间保存、变换信息 |
| 输入投影 `B` | 原始数值到节点的映射 | 各轮 Flow 通常固定，不随训练改变 |
| 循环矩阵 `W` | 节点之间的带符号连接权重 | 按实验设置冻结、慢速或正常更新 |
| 拓扑 `mask` | 允许存在的连接 | 固定结构；用掩码作用于稠密矩阵 |
| Write / 输入门 `p` | 元数据驱动的小型网络 | 控制每路输入的注入强度 |
| Route / 通路门 `g` | 任务信息和状态摘要驱动的网络 | 调节源节点组或源—目标块之间的传递 |
| Hold / 保持门 `a` | 状态更新系数 | 越小越保留旧状态，越大越吸收候选状态 |
| Readout / 读出 | 最终状态上的线性层 | 产生二分类 logit 或两个输出 logit |
| Salience | 事件相关性标量 | 第四轮研究从 key/query 学习事件匹配 |
| Selector | 任务条件节点选择表 | 第七轮在各角色内选择可活跃节点 |

“水流”是对信息传播的比喻，实际状态允许正负值，不是流体仿真。

### 早期模型

第 1～3 轮通常有 64 个节点，分为 4 块。上层决定输入强度，下层产生 4 个源块出边门；第 2 轮加入保持门，第 3 轮枚举三门组合。以下是概念形式，具体元数据和门的维数以各轮源码为准：

```text
p_t = sigmoid(Write(meta_t))
g_t = sigmoid(Route(task, block_mean(h_t)))
candidate_t = tanh((h_t * expand(g_t)) @ W + (x_t * p_t) @ B + meta_t @ C)
h_(t+1) = (1 - a_t) * h_t + a_t * candidate_t
logit = Readout(h_T)
```

其中 `C` 是早期模型的固定元数据投影。第 1 轮没有可学习保持门，更新系数固定为 0.2。第 3 轮的 `gIRH` 命名中，I/R/H 依次对应输入、通路、保持机制；关闭输入/通路机制是门全开，而非清零。

### 后期局部模块模型

第四轮 `routing.py` 引入 4 个具有固定角色的模块：来源 A、来源 B、目的 X、目的 Y。输入仅写入来源模块，两个线性头分别只读 X、Y；Route 输出 4×4 个源—目标块门。第五、六、七轮沿用或扩展这一思路。

```text
输入 A ──→ 来源 A ──┬──→ 目的 X ──→ headX
                   └──→ 目的 Y ──→ headY
输入 B ──→ 来源 B ──┬──→ 目的 X
                   └──→ 目的 Y

元数据 ──→ Write / Hold / Route
状态摘要 ─────────────────→ Route
```

各模块还允许内部循环连接；来源 A/B 之间不直接相连，目的 X/Y 之间不直接相连，目的模块不回传来源模块。允许的块内/块间位置再随机保留约 30% 的边。

## 3. 第七轮数据与任务

主要实现：[flow_mvp_v7/experiment.py](../flow_mvp_v7/experiment.py)。

`data(seed, n, task, steps=20)` 返回：

| 张量 | 形状 | 内容 |
|---|---|---|
| `x` | `[batch, steps, 2]` | `t=0` 写入数值 A，`t=1` 写入数值 B，其他位置为零 |
| `meta` | `[batch, steps, 7]` | 规则到达标记、规则符号、归一化时间、4 维任务 one-hot |
| `y` | `[batch, 2]` | 两个目标数值是否大于零 |

时间索引从 0 开始。任务身份从 `t=0` 可见，规则 R 从 `t=4` 起持续可见。每个 batch 必须只含一个任务，`Net.forward()` 对此有断言。

令 `sign(A)` 表示 `A > 0` 的二分类标签，四任务的目标为：

| 任务编号 | 名称 | R=0 的输出 | R=1 的输出 |
|---|---|---|---|
| 0 | A：按规则交换 | `(sign(A), sign(B))` | `(sign(B), sign(A))` |
| 1 | B：复制选中来源 | `(sign(A), sign(A))` | `(sign(B), sign(B))` |
| 2 | C：复制另一来源 | `(sign(B), sign(B))` | `(sign(A), sign(A))` |
| 3 | D：反向交换 | `(sign(B), sign(A))` | `(sign(A), sign(B))` |

任务名称 A/B 与输入数值 A/B 属于不同概念。训练顺序固定为任务 0→1→2→3，每阶段仅使用当前任务数据，无回放。

## 4. 第七轮模型与训练

### 模型结构

- `Net(n, mode)`：`n` 取 64、128、256、512、1024，4 个角色各有 `n/4` 个节点。
- `W`：`n × n` 可训练矩阵，初始化后谱范数为 0.9；训练中不持续约束该范数。
- `B` 与 `mask`：注册为固定 buffer。
- `write`：`7 → 8 → 2` 的 MLP，输出两路输入门。
- `hold`：`7 → 4` 线性层，更新系数为 `0.2 * sigmoid(...)`。
- `route`：`11 → 16 → 16` 的 MLP，输入为 7 维元数据和 4 个角色的状态摘要。
- `headX` / `headY`：各自只读取一个目的角色的最终状态。
- `selector`：形状 `[4, 4, n/4]`，分别对应任务、角色、角色内节点。

### dense 与 topk

`dense` 允许所有节点活跃。`topk` 按任务、按角色选分数最高的 16 个节点，总计 64 个，选择在同任务序列内固定，而非每步按输入重新决策。

前向使用硬掩码，反向使用 straight-through 软代理：

```text
soft = 16 * softmax(selector[task])
selection = hard_top16 + (soft - stop_gradient(soft))
```

只有 `topk` 且 `n > 64` 时选择表参与训练。64 节点下两个模式等价。Top-K 的状态摘要按 16 个活跃节点归一，循环项与读出另乘 `sqrt(n/64)`；因此两组的差异不仅是节点掩码。

代码仍使用稠密 `W` 和 `torch.einsum`，限制的是逻辑活跃节点，不是实际矩阵乘法规模。

### 训练协议

| 项目 | 第七轮设置 |
|---|---|
| 随机种子 | 11、22、33、44、55 |
| 每任务 train / validation / test | 512 / 256 / 1024 |
| 序列长度 / batch | 20 / 128 |
| 每任务 epoch | 30，即 120 次优化更新 |
| 每条持续学习流 | 4 阶段，共 480 次优化更新 |
| 优化器 / 学习率 | Adam / 0.003，每个新任务重建优化器 |
| 损失 | 最终两位输出的 BCEWithLogits |
| 梯度裁剪 | 全局范数上限 1.0 |
| BPTT | 每 5 步 `h.detach()`，仅训练时启用 |
| 模型选择 | 固定最后 epoch；验证分数仅记录 |
| 保存策略 | 每条流的最终模型；各阶段只保存分数和曲线 |

`detach()` 截断梯度，但保留状态数值。终点损失不再直接沿状态链传到早期输入；共享参数仍可由后期时间步获得梯度。

## 5. 指标解释

设 `M[s][t]` 为学完阶段 `s` 后在任务 `t` 上的准确率：

- **pair accuracy**：两个输出必须同时正确，第 4 轮局部路由及后续实验使用这一主指标。
- **plasticity**：`mean(M[t][t])`，第七轮四任务刚学完的平均准确率。
- **final_mean**：`mean(M[3])`，全部学习完成后的四任务平均准确率。
- **forgetting**：对前三个任务，取学会后各阶段最高分减最终分，再平均。
- **BWT**：对前三个任务，平均 `M[3][t] - M[t][t]`；负值表示相对刚学完时退步。
- **selection Jaccard**：不同任务的硬节点支持集交并比。
- **activation overlap**：平均绝对激活前 25% 节点的任务间交并比；正式分析额外剔除不大于 `1e-8` 的激活。
- **selectivity**：节点最强任务激活与其他任务平均激活的归一化差异；正式汇总仅统计有使用的节点。
- **route cosine**：结构存在的块边上，任务平均门值向量的余弦相似度。

`experiment.py` 保存基础诊断，`analyze.py` 的 `enrich()` 进一步计算去零激活后的指标。因此解释报告时应使用 `summary.json` / `all_metrics.json` 的正式汇总字段，而不是混用基础诊断里的同类字段。报告中的 `±` 是种子间样本标准差，不是置信区间。

## 6. 代码入口与数据流

| 文件 | 职责 |
|---|---|
| `flow_mvp/experiment.py` | 最小模型、RNN/MLP 基线及训练 |
| `flow_mvp_v2/experiment.py` | 标记输入、保持门及分布外测试 |
| `flow_mvp_v3/experiment.py` | 三门组合、W 更新和提示可靠性对照 |
| `flow_mvp_v4/routing.py` / `salience.py` | 局部目的路由、相关性学习分支 |
| `flow_mvp_v4/experiments.py` | 受控拓扑、另一套 Salience 与延迟规则探索 |
| `flow_mvp_v5/run.py` | 完整/10/5 步 BPTT × 冻结/慢速/正常 W 及 GRU |
| `flow_mvp_v6/run.py` | 预训练起点上的持续学习与独立适应对照 |
| `flow_mvp_v6/attribute.py` | 动态/静态/全开 Router 补充比较 |
| `flow_mvp_v7/experiment.py` | 容量和节点选择的 50 条学习流 |
| `flow_mvp_v7/joint_control.py` | 64 节点四任务交替训练参考 |
| `flow_mvp_v7/analyze.py` | 检查点复算、诊断、汇总及报告生成 |
| `flow_mvp_v7/plot.py` | 从汇总指标生成 `scaling.png` |
| `flow_mvp_v8/experiment.py` | 固定随机与可学习 Top-K 的严格配对归因实验（N=256） |
| `flow_mvp_v8/analyze.py` / `plot.py` | 第八轮重载核验、配对差值、报告与 `attribution.png` |
| `flow_mvp_v9/experiment.py` | 参数组保护与固定预算 replay 的遗忘定位实验 |
| `flow_mvp_v9/analyze.py` / `plot.py` | 第九轮全阶段核验、汇总、报告与 `forgetting.png` |
| `flow_mvp_v10/experiment.py` | A'B'C'D' 校准、验证选预算与 CL 基线 |
| `flow_mvp_v10/analyze.py` / `plot.py` | 第十轮状态复算、预算判定、报告与 `benchmark.png` |
| `flow_mvp_v11/experiment.py` | replay 比例 × 参数保护 × 任务顺序的 150 条流 |
| `flow_mvp_v11/analyze.py` / `plot.py` | 第十一轮复算、配对差值、报告与 `replay_matrix.png` |
| `flow_mvp_v12/experiment.py` | Flow/GRU/RNN 容量匹配与等预算 replay 训练 |
| `flow_mvp_v12/analyze.py` / `plot.py` | 第十二轮复算、所需 replay 插值、报告与 `architecture_replay.png` |
| `flow_mvp_v13/gradient_conflict.py` / `transplant.py` | 第十三轮梯度冲突与组件移植（复用第十一轮检查点） |
| `flow_mvp_v13/buffer_scan.py` / `analyze.py` / `plot.py` | 缓冲扫描、机制汇总与 `mechanism.png` |
| `flow_mvp_v14/experiment.py` | Write 五模式因果归因（frozen/const/full_bptt/aux） |
| `flow_mvp_v14/analyze.py` / `plot.py` | 第十四轮复算、判定、报告与 `write_attribution.png` |
| `flow_mvp_v15/experiment.py` | Flow-v2 正式基线（结构无 Write，输入常数注入） |
| `flow_mvp_v15/analyze.py` / `plot.py` | 第十五轮复算、基线判定、报告与 `flowv2_baseline.png` |
| `build_all_experiments.py` | 生成实验归档、索引和校验信息 |
| `.venv/` | 第八轮使用的 uv 管理 GPU 环境（WSL/Linux） |

第七轮的数据流：

```text
seed → data() → Net → 四阶段训练
                     ↓
              results/{mode}_{n}_{seed}.pt / .json
                     ↓
joint_control/ → analyze.py → summary.json / all_metrics.json / REPORT.md
                                   ↓
                                plot.py → scaling.png
```

各轮 `.pt` 格式不统一：早期常用 `state_dict`，第五～七轮主实验常用 `state`，第七轮混合训练直接保存 state dict。加载时应沿用对应轮次的构建函数和检查点键名。

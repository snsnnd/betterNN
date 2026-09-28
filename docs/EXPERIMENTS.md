# 二十四轮实验与结论

[返回项目首页](../README.md)

本文按研究问题整理归档结果。第 1～7 轮数值来自已有报告和结果文件；第 8、9 轮在 WSL GPU 运行，第 10 轮起在 CPU 并行运行。各轮任务、输入、预算和架构会变化，以下是研究脉络，不是跨轮排行榜。

## 1. 研究演进

| 轮次 | 问题 | 设计重点 | 主要观察 |
|---|---|---|---|
| [第1轮](../flow_mvp/REPORT.md) | 固定底层能否靠调制完成任务？ | 四路8步信号，选指定通道前4步求和 | 冻结底层＋调制器＋读出有效；严格只训练控制器不稳定 |
| [第2轮](../flow_mvp_v2/REPORT.md) | 能否处理随机有效时刻并保存信息？ | 有效标记、保持门、位置/长度/噪声迁移 | 选择和保持分别重要，部分种子出现通道近乎关闭 |
| [第3轮](../flow_mvp_v3/REPORT.md) | 三个门分别贡献多少？ | 完整三门消融，W学习率，提示噪声与缺失 | Route收益依赖种子；W训练改善ID；无提示存在信息缺失 |
| [第4轮](../flow_mvp_v4/REPORT.md) | 内容相关性和条件路由能否学到？ | Salience、局部目的模块、旧图审计及拓扑探索 | 辅助监督帮助Salience；动态Router在局部目的任务上有收益 |
| [第5轮](../flow_mvp_v5/REPORT.md) | 截短梯度窗口是否还能训练？ | BPTT完整/10/5步 × W冻结/慢速/正常及GRU | 正常W短窗口仍学会任务，冻结或慢速W在该预算下明显不足 |
| [第6轮](../flow_mvp_v6/REPORT.md) | 单任务能力能否转为持续学习？ | 预训练A后B→C→D，无回放 | 所有组明显遗忘，冻结W也不能保护控制器和读出 |
| [第7轮](../flow_mvp_v7/REPORT.md) | 扩容与显式节点选择是否改善保留？ | 5规模 × dense/Top-K × 5种子 | Top-K部分规模改善最终成绩，扩容收益不单调 |
| [第8轮](../flow_mvp_v8/REPORT.md) | Top-K 收益来自学习选择还是稀疏隔离？ | N=256，dense/固定随机/可学习Top-K严格配对 | 可学习未稳定优于固定随机；收益主要来自稀疏隔离 |
| [第9轮](../flow_mvp_v9/REPORT.md) | 持续学习到底是谁在遗忘？ | 冻结W/控制器/读出、只更新单组、固定预算replay | replay最有效；冻结控制器破坏旧任务 |
| [第10轮](../flow_mvp_v10/REPORT.md) | 任务是否独立可学？ | 新任务集A'B'C'D'、scratch/adapt单任务校准、验证选预算 | 全部任务约99.8%可学；65轮/阶段；新CL基线遗忘22.10个百分点 |
| [第11轮](../flow_mvp_v11/REPORT.md) | replay 比例与冻结 W 能否兼得可塑性和稳定性？ | 5种replay比例 × {all,freeze_W} × 3顺序 × 5种子 | replay几乎消除遗忘；freeze_W无叠加收益 |
| [第12轮](../flow_mvp_v12/REPORT.md) | 达到同样遗忘，Flow 需要多少 replay？ | Flow/GRU/RNN 容量匹配、4种比例 × 3顺序 × 5种子 | Flow 约6.7% replay 达遗忘≤3pp；GRU 25%仍未达到；RNN 未学会 |
| [第13轮](../flow_mvp_v13/REPORT.md) | replay 到底保护了什么？ | 梯度冲突、8种组件移植、缓冲M扫描 | 旧策略在Controller+W；replay降低冲突；Write从未被训练 |
| [第14轮](../flow_mvp_v14/REPORT.md) | Write 门到底有没有必要？ | frozen/常数写入/真训练/局部监督 5 模式 | 无可测差异，Write 可删除 |
| [第15轮](../flow_mvp_v15/REPORT.md) | 删除 Write 后能否复现基线？ | Flow-v2 结构删除 Write，重跑关键 replay 条件 | 复现成功，冻结 Flow-v2=W+Route+Hold+Readout |
| [第16轮](../flow_mvp_v16/REPORT.md) | 策略能否作为压缩的长期记忆？ | 固定带宽、扫描存储字节：样本 vs Route/Hold 锚点 | 策略锚点不能替代样本；遗忘瓶颈是数据锚定 |
| [第17轮](../flow_mvp_v17/REPORT.md) | 能力是否依赖严格时间递归？ | 预计算/迭代 Route 的四种接线 + 扫描验证 | CL 可并行化，长度外推依赖反馈闭环 |
| [第18轮](../flow_mvp_v18/REPORT.md) | 非线性闭环能压缩到什么程度？ | 块仿射 + 迭代重线性化（诊断，无训练） | 有预测器时深度 12～15 可恢复 Flow 外推 |
| [第19轮](../flow_mvp_v19/REPORT.md) | 可压缩性能否变成真实加速？ | 解析块 Jacobian 结构全量扫描 + GPU solver 基准（无训练） | 结构成立（rank≤4/块≤4L、scan tree 秩饱和~12、r=16 无损）；eager 实现无加速、长乘积溢出、PreRoute 长 T 发散 → 并行线冻结 |
| [第20轮](../flow_mvp_v20/REPORT.md) | 水从哪里进入、入口如何与动力学交互？ | 等输入能量下的 B 拓扑主族 / shared-pool overlap / `s_W` 扫描（5 seeds × 3 orders × r∈{0,12.5%}） | 任务侧无差异（H1 否、s_W 零结果）；动力学强效应（ρ_eff/G_max/lifetime）；overlap 增 W/Hold 冲突且 replay 下变差 |
| [第21轮](../flow_mvp_v21/REPORT.md) | B 能否学会把不同信息写进不同状态子空间？ | hybrid 信用分配（core 截断 / B 全 BPTT）+ 可训练 B + orth/overlap penalty + H3 机制链 + Full-BPTT 敏感性 | 5 步截断下 ∇B≡0；自发解耦弱、overlap penalty 强解耦但 hybrid 下不降遗忘；Full BPTT 下解耦消除 overlap 代价 |
| [第22轮](../flow_mvp_v22/REPORT.md) | 长程信用到底应该给谁：W、Route 还是 Hold？ | 8 arms 的 2³ 因子（`{W,Route,Hold}` 哪些拿 20 步 Full 信用，B 恒 Full）；逐组双通道梯度 + 10 seeds × 3 顺序 + 逐组 credit audit | **匹配口径下归因是零结果**（G=0.17pp，ME≤0.36pp，H1/H3/H4/H5 未通过）；V21 的 hybrid→Full gap 主要是 3 顺序 vs o0-only 的聚合错配（匹配后 1.11pp/5 seeds、0.17pp/10 seeds）；r=0 负控无效应；审计显示 W 的 5 步梯度与 20 步严重失配（cos≈0.33）但不转化功能差异 |
| [第24轮](../flow_mvp_v24/REPORT.md) | 单次瞬时写入是不是结构性缺陷？ | 等能量 write kernel（single/burst3/5/decay-fast/slow）× 延迟任务 T{20,40,80,160} + V23 Chain-select T=80 + 线性状态 probe | **时间结构在长 T 有作用且随 T 增强**（T160 +5.3pp）；**Chain-select K5 上 burst5/decay-slow 把 0.777→0.971 / 0.913→0.999**，single 已逐位回归；机制是末端可解码性而非扰动幅度 |
| [第23轮](../flow_mvp_v23/REPORT.md) | 造一个真正需要 core 长程信用的任务，还是证明不存在？ | Chain-select 压力任务（瞬态脉冲 + 长间隔 + 干扰 + 条件选择）× T{20,40,80,160} × core 窗口 K{5,10,20,Full} + sustained 对照 + 救援诊断（不改模型） | **没有找到长程硬案例**：T≥40 时 K=5 可达性 ≥ Full（T=80 xor 4/5 vs 3/5；T=160 2/3 vs 2/3）；唯一窗口效应是短 T 时序对齐（xor T20 K5 0/5）；主失败是"软选择平台/振荡"（Full 也不稳），K5 坏盆无法被 Full 微调救回（0.750→0.733） |

## 2. 关键证据

### 第1轮：概念验证成功，但读出不能随意冻结

固定底层、训练调制器和读出层的平均准确率为 **92.55%**，普通 RNN 为 **92.14%**。只训练两个控制器、连读出也保持随机冻结时为 **55.87%**。证据支持当前任务的可学习门控机制，不支持随机编码和随机读出下仅靠控制器即可稳定完成任务。

### 第2～3轮：写入、保持和底层学习相互影响

- 第2轮长干扰测试：标记调制模型 **63.54%**，加入保持门 **86.98%**，固定慢更新对照 **76.50%**。
- 第3轮常规测试：输入＋保持为 **84.83%**，完整三门为 **87.53%**；Route 的平均增益 2.70 个百分点主要由部分种子贡献。
- 第3轮正常更新 W 的三门组常规 **98.54%**，未见位置 **83.58%**；慢速 W 对应 **92.74% / 89.51%**，体现本配置下精度和迁移的取舍。
- 无提示时真实有效位置是独立随机变量，无法从信号唯一恢复。约 **70.49%** 的条件最优参考是已知生成分布的估计，不是模型自主发现重要性。

### 第4轮：必须区分两套研究路径

本目录混合保存了两套代码和产物：

| 路径 | 主要代码 | 对应资料 |
|---|---|---|
| 局部目的路由 / Salience辅助监督 | `routing.py`、`salience.py`、`audit_prior.py`、`make_report.py` | 当前 `REPORT.md`、`EXPERIMENT_PLAN.md`、`results/routing/` 及相关产物 |
| 受控拓扑 / Salience / 延迟规则探索 | `experiments.py`、`route_analysis.py`、`analyze.py`、`extra_analysis.py`、`verify.py` | `README.md`、`PLAN.md`、`results/topology/`、`delay/`、`delay_block/` 等 |

两条路径使用过相同的 `results/salience/` 目录，现有文件不能视为一套统一协议。具体表现是：当前 `results/salience/metrics.json` 使用 `study/mode/run_id` 字段，而 `make_report.py` 读取的是另一分支的 `kind/score_matrix` 字段。现有 `verification.txt` 也描述 36 模型分支，而当前 `verify.py` 对应探索分支。

当前报告中，任务损失独立训练 Salience 的常规/未见位置为 **79.11% / 59.28%**，额外匹配监督为 **98.23% / 95.27%**。局部目的路由任务中，动态路由 **76.22%**，静态 **59.29%**，动态＋慢速 W **91.41%**，GRU **99.35%**。这些应作为该报告对应分支的历史结果阅读，不能与另一分支的 `metrics.json` 逐行混配。

### 第5轮：短梯度窗口有效，不等于局部可塑性

| 配置 | 20步双输出全对率 | 40步双输出全对率 |
|---|---:|---:|
| W冻结，完整BPTT | 56.67% | 46.55% |
| W慢速更新，完整BPTT | 62.04% | 59.05% |
| W正常更新，完整BPTT | 99.10% | 99.87% |
| W正常更新，5步BPTT | 98.97% | 99.84% |
| GRU，5步BPTT | 37.29% | 31.22% |

这里保留前向状态、截断梯度，且规则在到达后持续可见，参数跨时间共享。没有实现局部学习规则，也没有由此解决持续学习。数据来自 3 个种子，未做等参数/等计算匹配和充分基线调参。

### 第6轮：新任务获取和旧任务保留是不同问题

持续学习从第五轮 full-BPTT 的预训练模型出发，三个 Flow 分支共享正常 W 训练得到的初始权重，此后分别冻结、慢速或正常更新 W。A 已预训练，只继续学习 B、C、D。

| 设置 | B/C/D刚学完平均 | 最终四任务平均 | 旧任务遗忘（百分点） |
|---|---:|---:|---:|
| frozen | 82.08% | 60.76% | 34.09 |
| slow | 82.22% | 60.64% | 34.39 |
| joint | 80.79% | 61.16% | 32.26 |
| GRU | 98.61% | 62.22% | 44.49 |

GRU 刚学完新任务更好，但最终同样明显遗忘。Flow 的遗忘较小伴随新任务获取较差，不能只看遗忘指标宣称优势。正常 W 组从相同预训练起点独立学习 C 达 **99.02%**，先学 B 再学 C 为 **62.21%**，说明存在明显顺序干扰。

[Router补充归因](../flow_mvp_v6/ATTRIBUTION_REPORT.md)属于独立研究：正常 W、5 步 BPTT 下，动态 Router 的 20/40 步成绩为 **98.97% / 99.84%**，全开组为 **90.06% / 65.46%**。这支持当前配置的 Router 收益，但不是持续学习结果。

## 3. 第七轮完整主结果

每组 5 个种子；双输出同时正确率，均值 ± 样本标准差。Top-K 每角色选择 16 个节点，逻辑活跃上限为 64。

| 节点 | 模式 | 刚学完新任务（%） | 最终平均（%） | 遗忘（百分点） |
|---|---|---:|---:|---:|
| 64 | dense | 75.28 ± 3.65 | 61.25 ± 1.60 | 19.28 ± 4.47 |
| 64 | topk | 75.28 ± 3.65 | 61.25 ± 1.60 | 19.28 ± 4.47 |
| 128 | dense | 78.38 ± 2.97 | 61.64 ± 1.40 | 22.75 ± 4.71 |
| 128 | topk | 87.13 ± 4.78 | 75.63 ± 6.19 | 15.34 ± 5.56 |
| 256 | dense | 76.68 ± 4.84 | 58.46 ± 4.15 | 25.36 ± 6.92 |
| 256 | topk | 92.72 ± 3.44 | 76.57 ± 4.80 | 21.54 ± 3.73 |
| 512 | dense | 77.65 ± 4.93 | 50.85 ± 10.61 | 35.74 ± 9.66 |
| 512 | topk | 91.82 ± 5.54 | 78.43 ± 4.53 | 18.06 ± 3.37 |
| 1024 | dense | 92.80 ± 10.76 | 57.81 ± 13.77 | 46.65 ± 6.07 |
| 1024 | topk | 86.91 ± 5.50 | 69.65 ± 7.51 | 23.37 ± 6.27 |

数据位置：[summary.json](../flow_mvp_v7/summary.json)、[results/](../flow_mvp_v7/results/)。

### 如何解释

1. **单纯扩容没有解决遗忘。** 1024 节点 dense 刚学完平均 92.80%，最终仅 57.81%。
2. **候选容量配合显式选择在部分规模有效。** 512 节点 Top-K 最终 78.43%，但 1024 节点降到 69.65%，不能认定单调扩容或普遍最优规模。
3. **分区并非全由学习产生。** 512 节点选择重叠从初始化 0.073 增至 0.163，1024 节点从 0.032 增至 0.118。任务私有随机选择表在训练前就制造了低重叠。
4. **混合训练参考也未充分拟合。** 64 节点、相同 480 次总更新、四任务交替训练为 **70.60 ± 4.33%**。它可访问所有任务训练集，不属于无回放持续学习。
5. **任务身份和角色是显式给定的。** 不能把结果表述为自主发现完整模块结构；Top-K 额外参数与缩放机制也参与了差异。

## 4. 第八轮：Top-K 机制归因

N=256，每角色选 16 个节点（共 64 个逻辑活跃），5 个种子，A→B→C→D、无回放。三组严格共享同一初始化、数据批序与更新次数；`fixed_topk` 与 `learned_topk` 前向数值相同，唯一差别是选择器是否接收梯度。

| 模式 | 活跃预算 | 刚学完（%） | 最终平均（%） | 遗忘（百分点） |
|---|---:|---:|---:|---:|
| dense | 256 | 76.68 ± 4.84 | 58.46 ± 4.15 | 25.36 ± 6.92 |
| fixed_topk | 64 | 91.61 ± 4.52 | 77.23 ± 7.78 | 19.23 ± 4.61 |
| learned_topk | 64 | 92.72 ± 3.44 | 76.57 ± 4.80 | 21.54 ± 3.73 |

- 核心配对（learned − fixed）final_mean 逐种子为 −5.66、+2.95、−6.98、−7.42、+13.82 个百分点，均值 **−0.66 ± 9.13**，正种子 2/5 → 预先约定判定为 H2/H0：**未观察到可学习选择的稳定优势**。
- learned 选择器确实明显移动（支持集 IoU 0.597，选择器 L2 位移 7.38），但任务选择重叠仅从 0.136 升到 0.180；学习没有产生更强的任务分区。
- 两个 Top-K 组都明显高于 dense（约 +18 个百分点），说明本协议下收益主要来自每任务随机支持集带来的稀疏隔离/容量，而不是 Router 学习。
- 设备一致性：本次 GPU 训练与第七轮 CPU 存档的离散成绩矩阵 5/5 相同，单条 CPU 复跑与存档逐位相同；内部浮点存在设备差异。
- 边界：dense 与 Top-K 的活跃预算、缩放和状态摘要归一不同；5 种子、单一任务顺序，阈值为工程判定而非统计显著性检验。

数据位置：[REPORT.md](../flow_mvp_v8/REPORT.md)、[summary.json](../flow_mvp_v8/summary.json)、[paired_differences.json](../flow_mvp_v8/paired_differences.json)。

## 5. 第九轮：持续学习遗忘定位

N=256 dense。先用全部参数学习 A（预训练后 A 任务平均 95.74%），再依次学习 B/C/D；七组分别保护不同参数组并加入固定预算 replay。5 个种子，另有 15 个独立适应对照。

| 组 | 可训练参数 | acquisition_BCD% | 最终平均% | 遗忘（百分点） |
|---|---:|---:|---:|---:|
| all | 66244 | 70.33 ± 6.57 | 58.46 ± 4.15 | 25.36 ± 6.92 |
| freeze_W | 708 | 74.92 ± 3.78 | 63.13 ± 2.32 | 22.96 ± 6.07 |
| freeze_ctrl | 65666 | 55.98 ± 4.26 | 34.99 ± 19.01 | 43.87 ± 20.51 |
| freeze_readout | 66114 | 72.68 ± 7.69 | 53.72 ± 9.51 | 33.39 ± 8.05 |
| ctrl_only | 578 | 75.84 ± 3.49 | 63.09 ± 1.19 | 23.63 ± 4.69 |
| readout_only | 130 | 47.45 ± 3.33 | 37.49 ± 16.21 | 29.52 ± 19.55 |
| replay | 66244 | 72.86 ± 3.47 | **69.46 ± 6.00** | **16.84 ± 4.64** |

- replay 最优：最终平均比 all 高 11.01 个百分点，遗忘低 8.52 个百分点，而当前任务每轮少看 25% 数据。
- 冻结控制器（W+读出仍可训练）最差：最终仅 34.99%，遗忘 43.87%，说明缺少可训练控制器时 W 与读出无法维持旧任务路径。
- 只保护读出并未保护旧任务（遗忘 33.39%）；只更新读出无法学会新任务（47.45%，独立适应对照 B/C/D 平均 74.50%）。
- freeze_W 与 ctrl_only 略优于 all，但配对标准差大，不能据此宣称冻结底层普遍更好。
- all 组的离散成绩矩阵与第八轮 dense 的 5/5 种子一致，确认保护框架没有改变基线路径。

数据位置：[REPORT.md](../flow_mvp_v9/REPORT.md)、[summary.json](../flow_mvp_v9/summary.json)、[paired_differences.json](../flow_mvp_v9/paired_differences.json)、[forgetting.png](../flow_mvp_v9/forgetting.png)。

## 6. 第十轮：持续学习 benchmark 校准

legacy 任务集中 D 与 A 互为规则逆映射，从 A 起点 adapt D 即使 150 轮也只有 83%（seed 22 卡在 54%），无法作为“任务可学”的干净基线。第十轮建立新 benchmark：A'=(A,B)/(B,A)（同legacy A）、B'=(A,A)/(B,B)（同legacy B）、C'=(A,B)/(A,A)、D'=(A,B)/(B,B)。

| 模式 | A' | B' | C' | D' |
|---|---:|---:|---:|---:|
| scratch | 99.90 ± 0.07 | 99.75 ± 0.31 | 99.75 ± 0.15 | 99.86 ± 0.16 |
| adapt from A' | 99.90 ± 0.07 | 99.86 ± 0.11 | 99.79 ± 0.22 | 99.80 ± 0.24 |

单位为最终测试 %（五种子）；所有任务 5/5 种子验证 ≥90%。验证集选预算：E90=48 轮、**E95=65 轮**，CL 固定 65 轮/阶段。

顺序 CL 基线（A'→B'→C'→D'，5 种子，固定最后状态）：

| 阶段结束 | A' | B' | C' | D' |
|---|---:|---:|---:|---:|
| 学完A' | 99.73 | 51.13 | 74.92 | 75.00 |
| 学完B' | 54.49 | 94.67 | 50.43 | 70.35 |
| 学完C' | 80.59 | 51.50 | 95.10 | 74.61 |
| 学完D' | 75.74 | 72.32 | 75.12 | 100.00 |

- acquisition **97.37 ± 3.24%**，最终平均 **80.80 ± 2.25%**，遗忘 **22.10 ± 5.86 个百分点**。
- 学习 B' 会重创 A'（54.49%），学习 C' 部分恢复 A' 但重创 B'，说明干扰主要发生在共享的策略通路上；第十一轮将用 replay × 参数保护矩阵直接检验。
- legacy 校准（含病态互补 D 的失败证据）保留在 [results_legacy_calibration/](../flow_mvp_v10/results_legacy_calibration/)。

数据位置：[REPORT.md](../flow_mvp_v10/REPORT.md)、[benchmark.json](../flow_mvp_v10/benchmark.json)、[benchmark.png](../flow_mvp_v10/benchmark.png)。

## 7. 第十一轮：Replay 比例 × 参数保护

固定每步 128 样本（replay 按比例替换当前数据）、A'B'C'D'、65 轮/阶段、3 个任务顺序、5 种子、每任务固定 32 个回放样本，共 150 条流。合并三个顺序的五个种子均值：

| 组 | replay% | acquisition% | 最终平均% | 遗忘（百分点） |
|---|---:|---:|---:|---:|
| all | 0 | 99.00 ± 2.10 | 77.21 ± 2.90 | 29.05 ± 5.99 |
| all | 6.25 | 99.47 ± 1.09 | 97.43 ± 2.58 | 3.14 ± 3.43 |
| all | 12.5 | 99.53 ± 0.82 | 99.05 ± 1.51 | 1.03 ± 1.97 |
| all | 25 | 99.47 ± 0.95 | 99.53 ± 0.39 | 0.39 ± 0.50 |
| all | 50 | 98.75 ± 2.19 | 99.58 ± 0.32 | 0.30 ± 0.37 |
| freeze_W | 0 | 97.68 ± 4.78 | 77.92 ± 3.28 | 26.35 ± 9.13 |
| freeze_W | 6.25 | 96.89 ± 5.45 | 93.18 ± 4.85 | 5.28 ± 4.59 |
| freeze_W | 12.5 | 96.51 ± 5.43 | 94.39 ± 5.27 | 3.58 ± 3.81 |
| freeze_W | 25 | 96.20 ± 5.45 | 95.98 ± 5.67 | 1.42 ± 2.00 |
| freeze_W | 50 | 95.12 ± 5.62 | 95.66 ± 5.92 | 0.61 ± 0.63 |

- 每步仅 8 个旧样本（6.25%）就把遗忘从 29.05 降到 3.14 个百分点；12.5% 时约 1 个百分点，25% 时 0.39。
- freeze_W − all 的配对最终平均差从 r=6.25% 起为 −4.25 ~ −4.92 个百分点（正差值 1/15），acquisition 也更低：冻结 W 没有稳定性收益，只损失可塑性。
- "W 稳定 + Controller 快速适应 + replay" 假说在本 benchmark 不成立，**replay 本身是主因**。
- 三个顺序结论一致（o1/o2 的 r=0 基线遗忘更高，为 32.3/32.7 个百分点，replay 后同样收敛到 <1%）。
- o0+r0+all 与第十轮基线离散矩阵 5/5 种子一致。

数据位置：[REPORT.md](../flow_mvp_v11/REPORT.md)、[summary.json](../flow_mvp_v11/summary.json)、[paired_differences.json](../flow_mvp_v11/paired_differences.json)、[replay_matrix.png](../flow_mvp_v11/replay_matrix.png)。

## 8. 第十二轮：Flow vs GRU vs RNN 等预算 replay 曲线

容量匹配：flow 66,244 参数（5 步窗口）、GRU 67,250（完整 BPTT）、tanh RNN 66,782（完整 BPTT）；A'B'C'D'、65 轮/阶段、每步 128 样本、32 样本/任务、replay 比例 0/6.25/12.5/25%、3 顺序 × 5 种子。

| 架构 | replay% | acquisition% | 最终平均% | 遗忘（百分点） |
|---|---:|---:|---:|---:|
| flow | 0 | 99.00 ± 2.10 | 77.21 ± 2.90 | 29.05 ± 5.99 |
| flow | 6.25 | 99.47 ± 1.09 | 97.43 ± 2.58 | 3.14 ± 3.43 |
| flow | 12.5 | 99.53 ± 0.82 | 99.05 ± 1.51 | 1.03 ± 1.97 |
| flow | 25 | 99.47 ± 0.95 | 99.53 ± 0.39 | 0.39 ± 0.50 |
| gru | 0 | 98.14 ± 0.48 | 75.99 ± 3.20 | 29.53 ± 4.35 |
| gru | 6.25 | 98.16 ± 0.51 | 92.40 ± 2.64 | 7.71 ± 3.43 |
| gru | 12.5 | 98.16 ± 0.46 | 93.25 ± 2.36 | 6.56 ± 3.24 |
| gru | 25 | 98.07 ± 0.46 | 93.84 ± 2.43 | 5.72 ± 3.25 |
| rnn | 0 | 60.43 ± 19.47 | 42.58 ± 19.06 | 26.13 ± 11.85 |
| rnn | 25 | 60.14 ± 21.22 | 48.82 ± 22.28 | 18.44 ± 10.49 |

| 架构 | 遗忘 ≤5pp 所需 replay | ≤3pp | ≤1pp |
|---|---:|---:|---:|
| flow | **5.80%** | **6.66%** | **13.06%** |
| gru | 未达到（25% 时仍 5.72pp） | 未达到 | 未达到 |
| rnn | 未达到（且未学会） | 未达到 | 未达到 |

- r=12.5% 时 flow − gru：最终平均 +5.79、遗忘 −5.53 个百分点；配对 15 对一致。
- vanilla RNN 在该预算下未学会任务（acquisition 60.43%），其遗忘数字不具解释力。
- flow 路径与第十一轮 all 组 o0 全部 20 条流矩阵逐位一致。
- 边界：架构默认设置不同（5 步截断 vs 完整 BPTT），未对 GRU/RNN 单独调参。

数据位置：[REPORT.md](../flow_mvp_v12/REPORT.md)、[summary.json](../flow_mvp_v12/summary.json)、[paired_differences.json](../flow_mvp_v12/paired_differences.json)、[architecture_replay.png](../flow_mvp_v12/architecture_replay.png)。

## 9. 第十三轮：replay 保护机制定位

全部基于第十一轮检查点；缓冲扫描为新增训练（flow、o0、r=12.5%、M∈{1,2,4,8,16,32}、5 种子）。

### 梯度冲突（最新任务 vs 旧任务，截断梯度）

| 参数组 | stage1 r0 | stage1 r12.5 | stage2 r0 | stage2 r12.5 | stage3 r0 | stage3 r12.5 |
|---|---:|---:|---:|---:|---:|---:|
| W | −0.441 | +0.151 | −0.098 | −0.333 | −0.399 | −0.296 |
| write | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| hold | −0.630 | +0.304 | −0.218 | +0.097 | −0.493 | +0.428 |
| route | −0.564 | −0.559 | +0.041 | −0.384 | −0.387 | −0.201 |
| readout | −0.160 | −0.125 | −0.343 | −0.409 | −0.560 | −0.066 |

- **Write 梯度恒为 0**：输入只在 t=0/1 写入，5 步截断的梯度窗口只覆盖最后 5 步，第七轮以来 Write 实际未被训练；完整 BPTT 对照下 Write 梯度非零。
- 使旧任务方向保持正相关所需 α* 中位数：W 0.81%、hold 0.15%、route 2.83%、readout 2.04%，都远小于实际使用的 12.5%。

### 组件移植（最终模型换回 stage0 组件）

r=0 时任务 A 基线 75.74%：换回 ctrl → 91.39%，换回 W → 82.09%，换回 readout → 75.74%（无影响），W+ctrl → 99.75%。**旧任务策略主要存放在 Controller，W 补足其余**。r=12.5% 模型本来四任务 ~99%，换回任何旧组件都只会破坏（ctrl 使 B/C/D 掉到 51/75/76）。

### 缓冲大小（r=12.5%）

| M/任务 | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| 遗忘（pp） | 24.78 | 22.16 | 13.80 | 6.80 | 4.14 | 0.73 |
| 最终平均% | 79.73 | 82.29 | 88.51 | 93.62 | 96.52 | 99.15 |

数据位置：[REPORT.md](../flow_mvp_v13/REPORT.md)、[summary.json](../flow_mvp_v13/summary.json)、[mechanism.png](../flow_mvp_v13/mechanism.png)。

## 10. 第十四轮：Write 因果归因

A'B'C'D'、Flow、o0、65 轮/阶段、5 种子、r∈{0,12.5%}；唯一变量是 Write。

| Write 模式 | r=0 最终% | r=0 遗忘 | r=12.5 最终% | r=12.5 遗忘 | ‖ΔWrite‖ |
|---|---:|---:|---:|---:|---:|
| frozen（现状随机固定） | 80.80 | 22.10 | 99.15 | 0.73 | 0.000 |
| const1（p≡1） | 81.61 | 22.52 | 99.04 | 0.91 | 0.000 |
| const05（p≡0.5） | 81.62 | 22.64 | 98.29 | 1.97 | 0.000 |
| full_bptt（真训练） | 81.56 | 22.54 | 98.28 | 1.86 | 1.869 |
| aux（局部监督） | 80.85 | 25.51 | 99.25 | 0.96 | 4.628 |

- 常数写入与随机固定 Write 无显著差异（配对差 ≤0.9 个百分点、方向 3/5 或 1/5）；真正被训练的 Write（ΔWrite≈1.9）与局部监督（ΔWrite≈4.6～6.3）也没有稳定收益。
- **判定：随机 task-conditioned 写入没有可测贡献，Write 门可从架构删除；输入注入直接用常数即可。** 架构简化为 W＋Route＋Hold＋Readout。
- aux 的 acquisition 达 99.99%，但遗忘反而更高（25.51）：局部监督加速学习但不帮助保留。
- frozen 与第十一轮逐位一致 10/10。

数据位置：[REPORT.md](../flow_mvp_v14/REPORT.md)、[summary.json](../flow_mvp_v14/summary.json)、[write_attribution.png](../flow_mvp_v14/write_attribution.png)。

## 11. 第十五轮：Flow-v2 正式基线

把 Write 从代码结构上删除（`FlowV2`，输入常数注入；用临时网络对齐 RNG 使初始化与 v11/v14 相同），r∈{0,6.25%,12.5%} × 3 顺序 × 5 种子。

| replay% | acquisition% | 最终平均% | 遗忘（百分点） | 旧 Flow 最终/遗忘 |
|---|---:|---:|---:|---:|---:|
| 0 | 99.38 | 77.48 | 29.19 | 77.21 / 29.05 |
| 6.25 | 99.47 | 97.03 | 3.70 | 97.43 / 3.14 |
| 12.5 | 99.49 | **99.44** | **0.52** | 99.05 / 1.03 |

- 删除 Write 后关键指标复现（r=0≈77%、6.25%≈97%、12.5%≈99%），且 o0 的 r∈{0,12.5%} 与第十四轮 const1 路径**逐位一致 10/10**。
- **正式冻结 Flow-v2 = W + Route + Hold + Readout**，后续实验以此为准。
- 结合第 13 轮移植：Controller 的恢复效果实际来自 **Route + Hold**（Write 未训练且无贡献），旧任务策略分布在 **Route＋Hold＋W**，Readout 不承载任务策略。

数据位置：[REPORT.md](../flow_mvp_v15/REPORT.md)、[summary.json](../flow_mvp_v15/summary.json)、[flowv2_baseline.png](../flow_mvp_v15/flowv2_baseline.png)。

## 12. 第十六轮：Sample replay vs Route/Hold Policy replay

Flow-v2、固定每步旧信息带宽 16 条（12.5%）、3 顺序 × 5 种子；唯一变量是旧知识的保存方式与存储预算（330 条流、1320 个阶段）。Policy 为模块级函数蒸馏（保存 q 与 sigmoid 前 logits，λ=0.5）。

| 方法 | 存储范围 | 遗忘范围（百分点） | 达到≤3pp 所需 |
|---|---|---:|---:|
| sample (x,meta,y) | 0.71～22.75 KB | 23.39 → **0.52** | **11.78 KB** |
| route_hold 锚点 | 0.73～23.25 KB | 24.20 → 26.04 | 未达到 |
| route 锚点 | 0.74～11.81 KB | 23.20 → 25.37 | 未达到 |
| hold 锚点 | 0.73～11.69 KB | 29.21（=no replay） | 未达到 |
| hybrid（锚点+极少量样本） | 3.62～26.09 KB | 19.24 → 9.55 | 未达到（同字节下劣于 sample） |
| none | 0 | 29.19 | — |

- **负结果：Route+Hold policy replay 不能替代原始样本。** 锚点从 0.7KB 加到 23KB 遗忘几乎不变；样本复现在 5.7KB 时遗忘 8.87、20.6KB 时 0.5。
- **漂移洞察**：sample 的 ‖ΔRoute‖ 最大（4.40）却保留最好；蒸馏把 ‖ΔRoute‖ 压到 1.25（λ=50）也不改善遗忘——**遗忘的瓶颈是状态动力学的数据锚定，而不是控制策略漂移**。
- λ 稳健性：0.5→50 使 Route 漂移减半，遗忘 18.16→17.77（o0/rh192），负结果对超参数稳健。
- none 与第十五轮 r=0 逐位一致 5/5。

数据位置：[REPORT.md](../flow_mvp_v16/REPORT.md)、[summary.json](../flow_mvp_v16/summary.json)、[policy_replay.png](../flow_mvp_v16/policy_replay.png)。

## 13. 第十七轮：可并行动力学可行性

四种接线同架构同参数量，A'B'C'D'、5 步截断、r∈{0,12.5%}、3 顺序 × 5 种子；新增 40 步外推与状态轨迹漂移。

| 模型 | r=12.5% 最终 | 遗忘 | 40步最终 | D_h 漂移（r=0） | 可 scan |
|---|---:|---:|---:|---:|---|
| A flowv2 | 99.44 ± 0.63 | 0.52 | 99.24 | 0.212 | ✗ |
| B linear（去 tanh） | 92.86 ± 4.22 | 5.54 | 76.91 | 4.361 | ✗ |
| C preroute（预计算 Route） | 93.88 ± 3.57 | 3.99 | 90.43 | 10.809 | ✓ |
| D iter（两遍迭代 Route） | 92.66 ± 4.13 | 5.20 | 73.74 | 4.287 | ✓ |

- **C（严格可 scan）在 CL 上落后 A 约 5.6pp 最终/3.5pp 遗忘，r=0 时几乎持平**；但 40 步外推落后 8.8pp，D 并未补回（73.74）。
- 去掉底网 tanh（B）同样有害：replay 效率 −6.6pp、40 步 −22pp——底层非线性是承重件。
- 扫描等价性：顺序 vs Hillis–Steele 仿射扫描最大偏差 2.2e-9（C）、2.6e-9（D 第二遍），数学性质成立。
- 状态轨迹漂移与遗忘不呈单调关系：A 的轨迹最稳定（0.212）但仍有 29pp 遗忘，扫描接线漂移大 20～50 倍、遗忘却相近；不能单独用漂移解释遗忘。

数据位置：[REPORT.md](../flow_mvp_v17/REPORT.md)、[summary.json](../flow_mvp_v17/summary.json)、[parallel_dynamics.png](../flow_mvp_v17/parallel_dynamics.png)、[scan_check.json](../flow_mvp_v17/scan_check.json)。

## 14. 第十八轮：block-affine 可压缩性（诊断）

不训练、不引入新架构：在第十五轮 Flow-v2（r=12.5%，o0，5 种子，stage3）上，把 T 等分为 B 块，每遍在入口估计 p 处取块函数与 Jacobian 方向导数，按仿射组合传播，K=0..3 次重线性化。初始估计为 `zero` 或第十七轮 `preroute` 模型轨迹。每任务 256 样本，评估 20/40 步。

参考：Flow 串行 acc20 99.17%、acc40 99.82%；PreRoute 模型 91.12%/86.37%。

| 初始估计 | L | K | 有效深度 | acc40% | E_h |
|---|---:|---:|---:|---:|---:|
| preroute | 1 | 2 | 3 | 97.34 | 0.071 |
| preroute | 2 | 2 | 6 | 98.14 | 0.018 |
| preroute | 4 | 2 | 12 | 99.26 | 0.0045 |
| preroute | 5 | 2 | 15 | 99.57 | 0.0025 |
| preroute | 2 | 3 | 8 | 99.79 | 0.0005 |
| zero | 10 | 2 | 30 | 98.75 | 0.004 |
| zero | 4 | 3 | 16 | 94.63 | 0.003 |

- **Oracle 自检精确为 0**：在真实入口线性化再组合恒等于原轨迹（恒等式），压缩损失来自近似入口。
- 有并行预测器时：**L=4～5 + 2 次重线性化、有效深度 12～15 即可把 40 步恢复到 99.3～99.6%**（Flow 99.82%）；无预测器需深度 20～40。
- 块 Jacobian σ_max≈10.25，扰动每块放大约 10 倍；迭代重线性化在实践中仍收敛（3 遍后 E_h ≈10⁻³～10⁻⁴）。
- 代价：每遍需要块 Jacobian 的作用；本轮证明的是"深度可压缩"，不是计算量已降低。另：上一稿块 Jacobi 方案（K<B 时末块入口恒零、梯度为零）作为失败记录保留在 `block_jacobi.py`。

数据位置：[REPORT.md](../flow_mvp_v18/REPORT.md)、[summary.json](../flow_mvp_v18/summary.json)、[compressibility.png](../flow_mvp_v18/compressibility.png)。

## 15. 第十九轮：并行求解器可行性（结构 + 0C 负结果）

不训练：5 seeds × 4 tasks × {20,40} 步 × L∈{1,2,4,5,10}，每配置 4 样本，先做解析算子结构全量扫描，再做 GPU solver 基准（T∈{40,128,512} × batch∈{1,8,32}，L=4，K∈{0,1}，zero 与 PreRoute 中心）。

### 结构结论

- 解析分解 `J_t = S_t + C_t`（C 来自 `pooled(h)→Route`，rank ≤4）与 autograd 一致到 float32 噪声底（单步 2e-8、块 1.7e-7）；S 保持 8/16 角色块支撑。
- 块修正秩均值 6.7～8.6（L=4，理论界 16）、8.6（L=10，界 40）；**scan tree 显著秩逐层饱和在 ~12**，未向 256 膨胀；截断 r=16 相对误差 ~9e-6（近无损），r=4 明显变差（~2e-2）。
- 回归：structured solver 与 v18 GS-JVP 在同一中心下 acc 完全一致；与 v18 存储 JSON（batch 256）max|Δacc|=0。

### GPU 基准（RTX 4060 Laptop，fp32，eager）

- **latency 未通过**：serial 本身 launch-bound（T=512: ~260–390ms）；GS-JVP 慢 6～13×；structured scan 慢 2～35× 且在 B≥32 时因跨块 Jacobian 乘积溢出 fp32 而失败；structured 顺序 apply 仅在 T≥128/batch=1 超过 serial（72ms vs 308ms），batch 8/32 被反超。
- **数值边界**：精确 affine scan 需要 `M_B···M_1` 长乘积，即使 zero 中心下也随块数增长溢出；**low rank ≠ well-conditioned**。
- **PreRoute 不能做长序列 predictor**：其状态 max|h| 在 T=40/128/512 = 1.3e3 / 2.1e10 / NaN（无 tanh 的线性动态长期不稳定）。
- 判定：未达 PLAN 预注册的 0C 门槛（latency≤serial、显存不炸、acc 达标），不进入 Phase 3；并行线冻结。复杂度核算表明 exact affine scan 的组合代价约为单步的 ~240 倍，在 N=256 规模下不划算。

数据位置：[REPORT.md](../flow_mvp_v19/REPORT.md)、[flow_mvp_v19/README.md](../flow_mvp_v19/README.md)、[results/summary.json](../flow_mvp_v19/results/summary.json)、[results/solver_bench.json](../flow_mvp_v19/results/solver_bench.json)、[results/solver_bench_zero.json](../flow_mvp_v19/results/solver_bench_zero.json)、[operator_structure.png](../flow_mvp_v19/operator_structure.png)、[solver_pareto.png](../flow_mvp_v19/solver_pareto.png)。

## 16. 第二十轮：B 输入拓扑 × 底网动力学

不改 Flow-v2 架构，只改 B 的支撑/符号/归一化与 W 初始尺度。统一协议：A'B'C'D'、5 seeds、65 轮/阶段、orders o0/o1/o2、r∈{0,12.5%}；每通道固定 `‖B_c‖₂=β=0.7√64=5.6`，`|B_ci|=β/√k`，固定种子 Rademacher 符号；先按 v15 正常构建模型再用独立 RNG 覆盖 B（RNG 流不串）。共 700 条流，完整性/矩阵复算误差 0。

### Phase A：主族 topology（single-random / single-central / multi8 / multi16 / distributed）

任务侧：单任务 final 全部 ≥0.994；CL r=0 遗忘全部 28–29pp，multi8 − single-random = **−1.01pp**、multi16 = **+0.02pp**（预注册门槛 −3pp 且 ≥4/5 → 未通过）；r=12.5% 主族 final 0.97–0.99。

动力学侧（单任务模型，任务×种子聚合）：

| 拓扑 | ρ_eff | γ4 | γ8 | 24 步衰减率 | ratio_end | G_max |
|---|---:|---:|---:|---:|---:|---:|
| single-random | 1.404 | 0.379 | 0.350 | −0.029 | 1.50 | 1.58 |
| single-central | 1.386 | 0.369 | 0.340 | −0.026 | 1.57 | 1.67 |
| multi8 | 1.271 | 0.230 | 0.195 | +0.058 | 0.56 | 0.92 |
| multi16 | 1.219 | 0.177 | 0.147 | +0.071 | 0.52 | 0.96 |
| distributed | 1.120 | 0.096 | 0.074 | +0.095 | 0.36 | 0.88 |

→ 输入越分散，扰动越不被放大、衰减越快（single-random 净放大且 τ½ 被 24 步窗口截尾；distributed τ½≈6.6 步）。**等能量拓扑只改变“河床动力学”，不改变任务表现。**

### Phase B：shared-pool overlap（两通道均可写 role0∪role1，每通道 8+8，α=0/0.5/1）

- 梯度冲突（stage0、o0、r=0，相对 α=0）：W **−0.219/−0.169**、Hold **−0.116/−0.111**（α=0.5/1，均 ≤−0.10），route 非单调（α=1 为 +0.136）→ H3 部分通过（2/3 组）。
- 功能面：r=0 遗忘 α=1 时 +2pp（未达 +3pp 辅助门槛）；r=12.5% 时 overlap 族 final **0.94/0.91/0.92**，明显低于主族 0.97–0.99。
- 寿命剂量效应：α=0/0.5/1 的 24 步衰减率 +0.056/+0.038/+0.016（共享越多，扰动越久）。

### Phase C：s_W = ‖W₀‖₂ ∈ {0.7, 0.9, 1.1} × {single-random, distributed}

任务与动力学指标全部无差异（干净零结果；训练把初始谱尺度洗掉）。

### 判定

- H1（multi-separate 降低 r=0 遗忘）：**未通过**（Δ≤1pp）。
- H2（distributed：可达性 ↑≥10%、τ½ ↓≥20%）：**可达性未通过**（随机输入 probe 全拓扑 r_eff≈1.1–1.2，不敏感）；**寿命方向支持**（基线 decay≤0，distributed +0.095）。
- H3（overlap→梯度冲突）：**部分通过**（W/Hold 通过，route 相反）。

数据位置：[REPORT.md](../flow_mvp_v20/REPORT.md)、[results/](../flow_mvp_v20/results/)、[input_topology.png](../flow_mvp_v20/input_topology.png)、[overlap_gradient.png](../flow_mvp_v20/overlap_gradient.png)、[sW_interaction.png](../flow_mvp_v20/sW_interaction.png)。

## 17. 第二十一轮：Adaptive Input Decoupling

不改架构，只把 B 变成可训练参数（写入域仅 role0∪role1，`‖B_eff‖₂=5.6`），研究输入解耦与持续学习的关系。协议：A'B'C'D'、5 seeds、o0/o1/o2、r∈{0,12.5%}；共 150 主实验流 + 100 robustness + 50 init 对照 + 120 Full-BPTT 敏感性 + fixed 回归。

### 两个方法学发现

- **5 步截断下 B 的梯度恒为 0**：输入在 t=0/1，而训练在 t=5/10/15 `detach()`，`∇B≡0`（与 v13 的 Write 同源；trunc 负控 5 seeds 实测 `|∇B|=0.0`）。
- **hybrid 信用分配**：core（W/Route/Hold/Readout）保持 5 步截断，B 单独用全 BPTT 梯度；同一参数快照、双 optimizer、分别 clip。Phase 0 硬检查：`max|g_core^hybrid − g_core^V20| = 0`、`|∇B|_hybrid≈3e-3`。fixed 臂与 V20 逐流矩阵差 = 0（12 条）。

### 判定结果（hybrid 主实验，3 顺序合并）

| arm | O_B init→final | r=12.5% 遗忘 | r=0 遗忘 |
|---|---|---|---|
| fixed-overlap | 1.00→1.00 | 9.04pp | 31.64pp |
| fixed-disjoint | 0→0 | 5.65pp | 29.81pp |
| learnable (λ=0) | 1.00→0.89 | 6.63pp | 30.78pp |
| learnable+overlap0.1 | 1.00→0.14 | 6.71pp | 30.01pp |
| learnable+overlap1 | 1.00→0.001 | 6.17pp | 29.64pp |
| learnable-random | 0.60→0.59 | 4.75pp | 31.39pp |

- **H1 复现通过（主条件）**：r=12.5% Δ=−3.39pp、4/5 负向；r=0 仅 −1.84pp。
- **H2a 未通过**：任务梯度自发降 O_B 到 init 的 0.83–0.91（0/5 ≤0.75）。
- **H2b 部分通过**：overlap penalty 强解耦（ΔO_B=−0.73/−0.86，两档方向一致）；orth 无作用；但解耦未带来显著遗忘下降（≤0.6pp）。
- **H3 机制链部分**：ρ(O_h,C∇)=0.59 达标，ρ(O_B,F)=0.41 边际，ρ(O_B,O_h)=0.21、ρ(C∇,F)=0.09 未达标。
- **V21D Full-BPTT 敏感性（40 流）**：r=12.5% 时 fixed-overlap 4.66pp → learnable/overlap0.1 **1.83/1.86pp**（fixed-disjoint 1.95）；r=0 各臂 ≈24pp。**B 拿到完整长程信用时，自适应解耦能消除 overlap 代价；hybrid 下不能——瓶颈是信用分配。**

数据位置：[REPORT.md](../flow_mvp_v21/REPORT.md)、[results/](../flow_mvp_v21/results/)、[decoupling.png](../flow_mvp_v21/decoupling.png)、[fullbptt_sensitivity.png](../flow_mvp_v21/fullbptt_sensitivity.png)。

## 18. 第二十二轮：Long-range Credit Attribution

不改架构、不动 B 的初始化/正则，只把 20 步完整信用**逐组**发给 `{W, Route, Hold}` 的 2³ 子集（B 恒可训练恒 Full）。每个 step 在同一参数快照上跑两条通道：截断通道（period=5）给 T 组、完整通道（20 步）给 Full 组，按组拼装梯度；双 optimizer、分别 clip，其余与 V21 hybrid 一致。协议：A'B'C'D'、10 seeds（11…110）、r=12.5% 跑 o0/o1/o2、r=0 跑 o0。臂：`B`（=V21 hybrid）、`B+W`、`B+R`、`B+H`、`B+W+R`、`B+W+H`、`B+R+H`、`All`。

### 协议核验

- `B` 臂与 V21A hybrid **逐位一致**（max|Δmatrix|=0，5 流）；首 batch 上完整通道 `autograd.grad` 与 V21D `backward` 梯度逐位相同。
- `All` 对 V21D 矩阵差 0.038（V21D 联合 clip vs V22 分别 clip，训练中 clip 会生效 max pre-clip norm≈1.9–2.5，1040 步放大的轨迹差），但遗忘均值 o0/5 seeds = 1.95 vs 1.83pp，结论层面等价。
- credit audit：min core `cos(g5,g20)` = 0.309（hold@init）<0.9，窗口确实改变信用方向。

### 判定结果（r=12.5%，3 顺序 × 10 seeds）

| arm | 遗忘 | | arm | 遗忘 |
|---|---:|---|---|---:|
| B (hybrid) | 5.40pp | | B+W+R | 5.00pp |
| B+W | 7.06pp | | B+W+H | 5.31pp |
| B+R | 5.58pp | | B+R+H | 6.42pp |
| B+H | 4.64pp | | All | 5.22pp |

- **归因零结果**：G=F(B)−F(All)=**0.17pp**；主效应 ME_W=−0.14、ME_R=+0.05、ME_H=+0.36pp（同向 5/10、4/10、5/10）；H1/H3/H4/H5 未通过，H2 以 |ME_W|≤1pp 通过（G≈0 时 closure 不可解释）。
- **V21 gap 重审**：V21 原口径 6.63pp（3 顺序）vs 1.83pp（仅 o0）= 表面 4.80pp；匹配同 5 seeds/3 顺序 = 1.11pp，10 seeds/3 顺序 = 0.17pp；分顺序 B−All = +1.08（o0，5/10）、+1.39（o1，6/10）、−1.94（o2，4/10）。**V21D 的“Full BPTT 消除 overlap 代价”不能外推到 3 顺序。**
- **r=0 负控干净**：所有 arm 24–25pp，|ME|≤0.4pp（H-neg 通过）——信用窗口只在 replay 下才有微弱作用。
- **credit audit**：训练后 `cos(g5,g20)`：W≈0.33（模长 6–10%）、hold≈0.50–0.65、route≈0.74–0.84、readout=1、B=0；窗口损失真实存在，但不转化为遗忘差异。
- o0 是唯一有弱信号的顺序（Hold ME +0.79pp、9/10；B−All +1.08pp），o1 同向不显著、o2 反转，说明是顺序×种子噪声主导。

数据位置：[REPORT.md](../flow_mvp_v22/REPORT.md)、[results/](../flow_mvp_v22/results/)、[attribution.png](../flow_mvp_v22/attribution.png)、[credit_audit.png](../flow_mvp_v22/credit_audit.png)。

## 19. 第二十三轮：Credit-Stress Benchmark

V22 在 A'B'C'D' 上没找到 core 长程信用需求，但旧任务允许最后窗口完成全部计算。V23 **不改模型、只造任务**：Chain-select——A@.05T、c1@.21T、distractor@.39T、B@.58T、c2@.76T 全部为瞬态脉冲，最后一步读出；任务为 `selected = A if c1==c2 else B`（`xor`）及其镜像（`xorsw`），两个输出全对才算对。B 恒可训练恒 Full，core `{W,route,hold}` 窗口 `K∈{5,10,20,Full}`。网格：transient 两任务 × T{20,40,80} × 4 K × 5 seeds + T=160（xor、3 seeds、K{5,20,Full}）+ sustained 对照（T=80，c1/c2 持续在 meta）。

### 判定结果

- **H0 校准通过**：Full 在 T=20/80 两任务均 ≥0.90（xor 0.955/0.912，xorsw 1.000/1.000）。
- **H1 未通过（无 T=80 硬案例）**：xor Full 0.912 vs K5 0.777（+13.5pp，但同向仅 2/5）；xorsw Full 1.000 vs K5 0.913（+8.6pp，3/5）。可达性（best-val≥0.9）xor：K5 **4/5** vs Full 3/5。
- **H2/H3/H4/H5 未通过**：T=20 的 gap（+20.5pp）反而大于 T=80（+13.5pp）；sustained 也没有让 K5 变好（Full 5/5 可达 vs K5 2/5 xor、4/5 xorsw）；T=160 Full 2/3 可达、K5 2/3、K20 3/3。
- **唯一干净的窗口效应在短 T 的时序对齐**：xor T=20 时 K5 可达 0/5（y0 卡 ~0.75）、K10 5/5、Full 4/5；K=5 末窗 [15..19] 只含 c2@15、缺 B@12。
- **主失败模式是"软选择平台/周期振荡"**：y0≈0.75–0.94，Full 同样中招（T=40/80 可达 3/5；T=160 出现 val 1.0→final 0.38 崩溃），所以 final 的 K 差异不能归因于 credit horizon。
- **救援诊断**：T=20 xor 的 K=5 失败流续训 Full 100 epochs 无一条被救回（0.750→0.733）——坏盆是吸引子；失败是轨迹/盆地选择，不是表示能力或信用不足。

数据位置：[REPORT.md](../flow_mvp_v23/REPORT.md)、[results/](../flow_mvp_v23/results/)、[credit_stress.png](../flow_mvp_v23/credit_stress.png)。

## 20. 第二十四轮：Input Write Dynamics

V23 留下的最大疑点：单次瞬时写入是否本身脆弱？V24 **不改模型结构**，只把写入改成等能量的因果 FIR trace：`z_t = Σ_k a_k B x_{t-k}`，`Σa_k²=1`（外部输入仍只出现一次）。kernels：`single=[1]`、`burst3/5` 均匀、`decay-fast/slow ∝ 0.5^k/0.85^k`。Phase A：固定 B（V20 fixed-disjoint）、延迟任务 A'B'C'D'、T{20,40,80,160}、65 epochs、5 kernels × 4 tasks × 5 seeds；Phase B：V23 Chain-select T=80、保持 learnable B hybrid / core K=5、500 epochs，只换 kernel。

### 判定结果

- **回归**：`single@T20` 与 V20 fixed-disjoint singles **逐位一致**（20/20，max|Δfinal|=0）；Phase B 的 `single` 与 V23 同 cell **逐位一致**（10/10）。
- **H1 未通过（延迟任务 T=80 严格门槛）**：T=80 最优 `burst5` +3.5pp（17/20 同向），未到 +5pp；T≤40 无差异。
- **H2 通过**：gap 随 T 单调增长：−0.0 / +1.1 / +3.5 / **+5.3pp**（T=20/40/80/160）。
- **H3 通过（关键）**：Chain-select T=80、core 仍 K=5、B 仍 hybrid 时：`single` 0.777（xor）/0.913（xorsw）→ `burst5` **0.971** / `decay-slow` **0.999**，且 single 的崩溃 seed 全部救回（5/5 可达）。**V23 的 soft-selection 失败很大程度源于单次瞬时写入。**
- **H4 未通过，机制被 probe 改写**：||Δh|| retention 与 acc 不同向（全网格 r=0.08，T≥80 子集 r=0.42）；线性 probe（前 512 拟合/后 512 评估）在 T/2 与 acc 负相关（r=−0.47）、T−1 弱正相关（r=+0.46）——kernel 改善的是**末端状态几何/可读出性**，不是简单把扰动留得更久。
- 偏好 kernel 依赖任务（xor→burst5、xorsw→decay-slow）→ 证明“写入时间策略”是一个真实自由度，下一步应做**可学习 scheduler（V24B）**而不是继续固定 kernel。

数据位置：[REPORT.md](../flow_mvp_v24/REPORT.md)、[results/](../flow_mvp_v24/results/)、[write_dynamics.png](../flow_mvp_v24/write_dynamics.png)、[chain_kernels.png](../flow_mvp_v24/chain_kernels.png)。

## 21. 已保存的内容

| 轮次 | Python文件 | `.pt`文件 | 保存内容说明 |
|---|---:|---:|---|
| 1 | 4 | 24 | 8配置 × 3种子的模型 |
| 2 | 5 | 30 | 9主配置及固定慢更新对照 |
| 3 | 6 | 75 | 15配置 × 5种子 |
| 4 | 12 | 122 | 两分支新模型、旧模型及重复历史副本 |
| 5 | 5 | 36 | 4模型设置 × 3窗口 × 3种子 |
| 6 | 7 | 63 | 6预训练起点＋36阶段＋9独立控制＋12归因模型 |
| 7 | 5 | 55 | 50条流最终模型＋5个混合训练模型 |
| 8 | 4 | 16 | 3模式 × 5种子最终模型＋1条CPU一致性检查 |
| 9 | 3 | 125 | 5预训练＋105阶段模型＋15适应对照 |
| 10 | 3 | 100 | 校准最优/最终状态＋CL阶段模型 |
| 11 | 3 | 600 | 150条流×每流4个阶段检查点 |
| 12 | 3 | 720 | 180条流×每流4个阶段检查点 |
| 13 | 6 | 120 | 缓冲扫描30条流×4阶段（冲突/移植为分析产物） |
| 14 | 3 | 200 | 50条流×每流4个阶段检查点 |
| 15 | 3 | 180 | 45条流×每流4个阶段检查点 |
| 16 | 3 | 1360 | 330条流×4阶段＋λ稳健性检查 |
| 17 | 4 | 480 | 120条流×4阶段（40步外推与 scan 验证为附加产物） |
| 18 | 4 | 0 | 诊断：无训练，仅 JSON 结果与图 |
| 19 | 4 | 0 | 诊断：结构扫描 + GPU 基准，无训练、无检查点 |
| 20 | 4 | 2084 | 3 阶段 × 5 seeds：CL 逐阶段检查点 + 单任务最终模型 |
| 21 | 4 | 1144 | V21A/B/C/D + fixed 回归：CL 逐阶段检查点 + 单任务模型 |
| 22 | 4 | 488 | 2³ 信用归因：80 条 o0/r=12.5% 流逐阶段检查点 + 160 单任务模型 + 8 条回归流；credit audit 只读 V21 检查点 |
| 23 | 5 | 0 | 信用压力任务：149 条单任务流 + 救援诊断，无检查点（只存 JSON） |
| 24 | 3 | 0 | 写入时间结构：430 条流（Phase A 400 + Phase B 30），无检查点（只存 JSON） |
| 原归档合计 | 44 | 405 | 第1～7轮文件数，不是独立实验次数；第八～十八轮另计 |

训练、验证和测试输入大部分未单独保存为张量文件，而是由代码和种子生成。第七轮共有 200 个任务阶段，但仅有 50 个顺序学习最终检查点；第八轮只保存最终模型；第九轮保存每个阶段的模型，可重载复核。

## 22. 当前研究状态

第二十一轮发现 `∇B≡0` 并用 hybrid 信用分配（core 截断 / B 全 BPTT）训练 B，当时的结论是“Full joint 长程信用几乎消除 overlap 代价（6.63→1.83pp）”。**第二十二轮用匹配协议（同 10 seeds × 3 顺序）重审并修正了这个结论**：

- V21 的 gap 主要是**聚合错配**：6.63pp 是 3 顺序 hybrid、1.83pp 是仅 o0 的 Full；匹配后 5 seeds/3 顺序 gap 只有 1.11pp，10 seeds/3 顺序 0.17pp，分顺序符号翻转（o2 Full 反而更差）。V21D 的 o0-only 结论不可外推。
- 预注册的 2³ 归因是**零结果**：给 `{W,Route,Hold}` 的任何子集（含 All）打开 20 步信用，r=12.5% 遗忘都无稳定变化（G=0.17pp、|ME|≤0.36pp）；r=0 负控同样无效应。
- credit audit 说明“梯度看不远”是真的（训练后 W 的 cos(g5,g20)≈0.33、模长只剩 6–10%），但**不构成这些任务的遗忘瓶颈**：B 可以把长程信息写进状态，core 只需在尾部窗口完成读出。
- 唯一稳健的长程信用需求仍是 **B 的完整信用**（V21：截断下 `∇B≡0`，B 完全学不到），hybrid 的双通道已经用很便宜的方式解决。

**第二十三轮按 V22 的建议造了 Credit-Stress Benchmark（Chain-select：瞬态脉冲 + 长间隔 + 干扰 + 条件选择，T 到 160）来主动寻找 core 窗口的硬案例，结果仍未找到**：

- T≥40 时 K=5 的可达性 ≥ Full（T=40：5/5 vs 3/5；T=80 xor：4/5 vs 3/5；T=160：2/3 vs 2/3）；
- 唯一干净的窗口效应是**短 T 的脉冲-边界对齐**（xor T=20：K5 0/5、K10 5/5），不是 horizon；
- 主失败模式是“软选择平台/周期振荡”，Full 同样不稳定；K5 的坏盆无法被 Full 微调救回（救援诊断 0.750→0.733）；
- sustained 对照（指令持续可见）方向不一致，说明瓶颈不是命令可见性。

因此 Flow-v3 的方向维持 V22 的判断并加强：**不要为 core 设计 eligibility trace / synthetic gradient 等长程信用机制**；core 用 5 步窗口即可，唯一需要跨窗口信用的是 B（输入写入），hybrid 双通道已解决。并行/solver 线继续冻结（见 [HANDOVER §9](../HANDOVER.md)）。

**第二十四轮换了一个自由度：不改模型、只改“怎么随时间写入”（等能量 FIR kernel），发现这是真实且长期被忽略的变量**：

- 固定 B 的延迟任务上，写入时间结构的收益随 T 增长（T=20/40 无差异；T=80 +3.5pp；**T=160 +5.3pp**，16/20 同向）；
- 在 V23 的 Chain-select（T=80、core 仍 K=5、B 仍 hybrid）上效果很大：`single` 0.777/0.913 → `burst5` **0.971** / `decay-slow` **0.999**，并把 single 的崩溃 seed 全部救回；
- 机制不是“扰动留得更久”（||Δh|| 振幅与 acc 不同向），更像**末端状态可解码性**（probe T−1 r=+0.46）；
- 偏好 kernel 依赖任务（xor→burst5、xorsw→decay-slow），因此下一步不是继续固定 kernel，而是 **V24B：可学习 write gate `w_t` / scheduler（只看 `(x_t, meta_t)`，保持并行性）**，再验证 `λ_t`（记多久）与 `α_t`（写到哪）三个自由度里谁真正有用。

当前路线：**Where（B 支撑/解耦，V20/V21 已答）→ When/How much（V24 正在做）→ How long（λ_t，V24C）→ Flow-v3**。core 长程信用与 eligibility trace 方向保持冻结。

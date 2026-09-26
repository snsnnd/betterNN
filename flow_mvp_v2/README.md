# 大网＋调制器：第二轮实验

本轮承接第一轮的关键未决问题：控制器能否根据有效标记处理随机出现的信息，而不只是记住固定的“前四步开、后四步关”？当干扰延长时，选对的信息能否留住？

## 运行

Python 3.10+，CPU 即可，无外部数据。

```bash
pip install -r requirements.txt
python predict.py
```

复现九组主实验及一组固定慢更新对照：

```bash
python experiment.py --epochs 100 --seeds 11 22 33 --out results_reproduced
python experiment.py --variants marker_slowleak --epochs 100 --seeds 11 22 33 --out results_slowleak_reproduced
python verify.py --out results_reproduced
python verify.py --out results_slowleak_reproduced
```

只跑核心保持门模型：

```bash
python experiment.py --variants marker_memory --seeds 11 --out quick_run
python predict.py --checkpoint quick_run/marker_memory_seed11.pt
```

原始结果在 `results/` 和 `results_slowleak/`，报告见 `REPORT.md`。绘图命令 `python make_plot.py` 读取这两个目录。

## 任务

输入四路12步独立随机信号、四维任务指令q、每步有效标记m和归一化时间。每例恰好两个有效时刻，训练时随机落在前六步。最后回答：q指定通道的两个有效值相加是否大于零？有效标记独立于信号值及答案，不是标签泄漏。

所有模型看到同样的输入变量。标记是外部给定的“哪些时刻有效”的提示；本实验不检验从无标记信号中自主识别重要性。已知任务规则可直接求和得到100%准确率；神经网络实验研究的是从样本学习这条规则及其泛化。

## 结构

64个状态节点、四块、20%固定随机连接；固定编码B、固定元数据投影C。上层输出四路注入强度p，下层按四块状态均值和q输出出边门控g。

```
meta_t = [q, t/(T-1), m_t]
p_t = sigmoid(Upper(meta_t))
g_t = sigmoid(Lower([q, block_mean(h_t)]))
candidate_t = tanh((h_t * expand(g_t)) @ W
                   + (signal_t * p_t) @ B + meta_t @ C)
h_(t+1) = (1-a_t)*h_t + a_t*candidate_t
logit = Linear(h_T)
```

原调制模型a恒为0.2。新增保持门模型a=0.2*sigmoid(Linear(meta_t))，仅新增7个可训练参数：小a少更新、保留旧状态，大a吸收更多候选状态。该门未硬编码为有效标记，必须学习。初始化a=0.1，因此加入a恒为0.1的固定慢更新对照排除初始化速度差异。

上层不看信号值，下层只看已有状态摘要与任务。输出层只能看最终状态，没有原始输入旁路。元数据通过C直接进入候选状态，各Flow组都保留此路径。故即使上层不看标记，底层仍可能利用标记。

## 对照

| 名称 | 含义 |
|---|---|
| readout_only | 冻结随机网络，只训练读出 |
| clock_flow | 上层只能看q与时间；底层仍看有效标记 |
| marker_upper | 上层看标记，训练上层与读出，下层全开 |
| marker_flow | 上层看标记，训练上下层与读出，固定a=0.2 |
| marker_memory | 在marker_flow上加可学习保持门 |
| marker_slowleak | 与marker_flow相同，但固定a=0.1 |
| slow_backbone | marker_flow加W慢速微调，不包含保持门 |
| controller_only | 严格仅训练上下层，读出随机冻结；不包含保持门 |
| gru_small | 8隐藏单元GRU，489个可训练参数 |
| gru64 | 64隐藏单元GRU，14657个可训练参数 |

主模型marker_flow为457个可训练参数，保持门模型为464个。仍有底层固定权重和缓冲区；不能把可训练参数数目当作完整模型大小。W微调参数统计包括被掩码遮住的位置。基线没有充分超参数搜索，结果不构成架构上限比较。

## 五种独立测试分布

| 条件 | 长度 | 两个有效位置 | 无效信号标准差 | 检验 |
|---|---:|---|---:|---|
| id12 | 12 | 前6步随机 | 1 | 同分布学习 |
| shift12 | 12 | 后6步随机 | 1 | 未见有效位置 |
| random12 | 12 | 全12步随机 | 1 | 任意位置及间隔 |
| delay24 | 24 | 前6步随机 | 1 | 更长干扰和记忆保持 |
| noise12 | 12 | 前6步随机 | 3 | 更强干扰 |

时间输入归一化到[0,1]，因此delay24也改变了前六步的归一化时间值；它是组合分布外压力测试，不能把全部变化归因于纯记忆长度。shift12的有效位置虽然未在训练出现，但平均距输出更近，因此难度并非单向提高。

各测试条件使用独立样本。同种子下，各模型使用相同数据、批次顺序；Flow初始权重相同。训练3072、验证768、每种测试4096；三种子11/22/33、100轮、batch256，Adam 0.003，W微调0.00015。只按同分布验证集选checkpoint，测试不参与模型选择。

## 可信度检查

`verify.py --data-only` 核验标签、有效标记数和范围、确定性生成、控制器梯度和冻结W。`verify.py` 还会逐个重新加载检查点，重算五种测试准确率，并对比冻结参数与初始化。

上层全开、下层全开、保持门固定、标记清零，是对已训练模型的干预，会引入分布变化，不能代替重新训练消融。标记清零后保持原始标签，专门检查对提示的依赖。

三随机种子的样本标准差不是置信区间。不因单一最佳种子下结论。第一轮和本轮任务、数据量、输入均已改变，不能把两轮准确率直接作升降比较。

## 限制

这里的“水流”是带符号神经状态，不是流体守恒模型。软门控不等于真实稀疏计算或降低带宽，代码仍使用稠密矩阵乘法。所有Flow状态通过凸组合和tanh保持有界，但未证明全局收敛；状态依赖门控不能直接套固定线性系统稳定性结论。

未验证语言任务、无提示重要性识别、更多有效事件、训练未见任务、真实传感器数据、无限长序列；未实现硬带宽预算、脉冲频率和相位。训练计时只反映本次CPU运行，不是严格性能基准。

## 文件

- `experiment.py`：完整数据、模型、训练、测试和干预。
- `EXPERIMENT_PLAN.md`：预设实验和固定慢更新对照的添加时点。
- `verify.py`：语义及检查点核验。
- `predict.py`：已有模型推理演示。
- `make_plot.py`：实际指标绘图。
- `results/`、`results_slowleak/`：配置、历史、JSON/CSV指标和检查点。
- `REPORT.md`：实测结论。

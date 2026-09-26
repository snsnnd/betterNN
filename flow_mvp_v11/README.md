# 第十一轮：Replay 比例 × {all, freeze_W}

回答：**底层 W 稳定、只让控制器快速适应、再用少量旧经验防漂移，能否同时得到可塑性与稳定性？**

- 设计见 [PLAN.md](PLAN.md)；结果见 [REPORT.md](REPORT.md)（由 `analyze.py` 生成）。
- 图：`replay_matrix.png`（比例曲线 + Pareto）。

## 结果摘要（150 条流）

- replay 大幅解决遗忘：all 从 r=0 的最终 77.21%/遗忘 29.05 个百分点，变为 r=6.25% 的 97.43%/3.14、r=12.5% 的 99.05%/1.03、r=25% 的 99.53%/0.39。
- **freeze_W 没有叠加收益**：各比例下 freeze_W 的最终平均都比 all 低 3.5～4.9 个百分点（配对 1/15 为正），遗忘反而略高。
- 结论：**稳定性瓶颈主要由 replay 解决，冻结 W 只损失可塑性**；"慢河床 + 快闸门"假说在本 benchmark 上不成立。
- o0+r0+all 与第十轮基线 5/5 种子一致。

## 矩阵

- 任务集：第十轮 A'B'C'D' benchmark，65 轮/阶段。
- 顺序：`o0`=A'B'C'D'、`o1`=D'C'B'A'、`o2`=B'D'A'C'。
- 组：`all`（全参数）、`freeze_W`（首任务后冻结 W）。
- replay 比例：0 / 6.25% / 12.5% / 25% / 50%，每步固定 128 样本，按比例替换当前数据。
- 5 个种子，共 150 条流。

## 运行（CPU）

```bash
cd flow_mvp_v11
# 三个顺序可并行；示例单顺序：
/mnt/d/betterNN/.venv/bin/python experiment.py --orders o0 --out results --device cpu
/mnt/d/betterNN/.venv/bin/python analyze.py
/mnt/d/betterNN/.venv/bin/python plot.py
```

冒烟：`--epochs 2 --seeds 11 --ratios 0 .25 --orders o0 --arms all freeze_W`。

## 文件

- `experiment.py`：顺序/回放/冻结训练与阶段检查点。
- `analyze.py`：全量复算、汇总、配对差值与报告。
- `plot.py`：`replay_matrix.png`。
- `results/`：150 条流的 JSON 与阶段检查点。

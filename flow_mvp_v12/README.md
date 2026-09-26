# 第十二轮：Flow vs GRU vs RNN 等预算 replay 曲线

回答：**达到同样的遗忘水平，Flow 架构需要多少 replay？容量匹配的 GRU/普通 RNN 相比是否更省历史信息？**

- 设计见 [PLAN.md](PLAN.md)；结果见 [REPORT.md](REPORT.md)（由 `analyze.py` 生成）。
- 图：`architecture_replay.png`（曲线 + 所需 replay 比例）。

## 结果摘要（180 条流）

| 架构 | 参数 | 达到遗忘≤5pp | ≤3pp | ≤1pp |
|---|---:|---:|---:|---:|
| flow | 66,244 | **5.80%** | **6.66%** | **13.06%** |
| gru | 67,250 | 未达到（25% 时仍 5.72pp） | 未达到 | 未达到 |
| rnn | 66,782 | 未达到（且未学会：acquisition 60.4%） | 未达到 | 未达到 |

- r=12.5% 时 flow − gru：最终平均 **+5.79**、遗忘 **−5.53** 个百分点。
- flow 路径与第十一轮 all 组 o0 全部 20 条流矩阵逐位一致。
- 结论：**同等 replay 预算下，Flow 达到相同遗忘水平所需历史信息显著少于容量匹配的 GRU/RNN**。

## 矩阵

- A'B'C'D'，65 轮/阶段，每步固定 128 样本，replay 比例 0/6.25/12.5/25%，32 样本/任务。
- 架构：`flow`（N=256，66k 参数，5 步窗口）、`gru`（H=144，67k）、`rnn`（H=252，67k，完整 BPTT）。
- 顺序 o0/o1/o2 × 5 种子 = 180 条流。

## 运行（CPU，可三顺序并行）

```bash
cd flow_mvp_v12
for o in o0 o1 o2; do
  /mnt/d/betterNN/.venv/bin/python experiment.py --orders $o --out results --device cpu > logs/$o.log 2>&1 &
done
wait
/mnt/d/betterNN/.venv/bin/python analyze.py
/mnt/d/betterNN/.venv/bin/python plot.py
```

冒烟：`--epochs 2 --seeds 11 --ratios 0 .25 --orders o0`。

## 文件

- `experiment.py`：三种架构、replay 训练与阶段检查点。
- `analyze.py`：全量复算、所需 replay 比例插值、报告。
- `plot.py`：`architecture_replay.png`。
- `results/`：180 条流的 JSON 与检查点。

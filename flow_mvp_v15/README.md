# 第十五轮：Flow-v2 正式基线

把 Write 从代码结构上彻底删除，建立正式基线 **Flow-v2 = W + Route + Hold + Readout**（输入常数注入）。

- 设计见 [PLAN.md](PLAN.md)；完整结果见 [REPORT.md](REPORT.md)。
- 图：`flowv2_baseline.png`。

## 结果摘要（45 条流，3 顺序 × 5 种子）

| replay% | acquisition% | 最终平均% | 遗忘（百分点） |
|---|---:|---:|---:|
| 0 | 99.38 | 77.48 | 29.19 |
| 6.25 | 99.47 | 97.03 | 3.70 |
| 12.5 | 99.49 | **99.44** | **0.52** |

与旧 Flow（含未训练的 Write）几乎一致（r=0: 77.21→77.48；6.25%: 97.43→97.03；12.5%: 99.05→99.44），且 o0 的 r∈{0,12.5%} 与第十四轮 const1 逐位一致 10/10。

**判定：复核标准成立，正式冻结 Flow-v2 为后续实验的基线架构。** 结合第 13 轮移植，旧任务策略分布在 **Route + Hold + W**，Readout 不承载任务策略；Write 无贡献。

## 运行（CPU，3 顺序并行）

```bash
cd flow_mvp_v15
for o in o0 o1 o2; do
  /mnt/d/betterNN/.venv/bin/python experiment.py --orders $o --ratios 0 .0625 .125 --out results --device cpu > logs/$o.log 2>&1 &
done
wait
/mnt/d/betterNN/.venv/bin/python analyze.py
/mnt/d/betterNN/.venv/bin/python plot.py
```

## 文件

- `experiment.py`：Flow-v2 模型（无 `self.write`）、RNG 对齐与 replay 训练。
- `analyze.py`：全量复算、与 v12/v14 的核对、判定。
- `plot.py`：`flowv2_baseline.png`。
- `results/`：45 条流 JSON 与 180 个阶段检查点。

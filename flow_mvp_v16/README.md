# 第十六轮：Sample replay vs Route/Hold Policy replay

固定 Flow-v2，只改"旧知识怎么保存"，以**存储字节数**为横轴检验：Route/Hold 策略锚点能否比原始样本更省存储地防止遗忘。

- 设计见 [PLAN.md](PLAN.md)；完整结果见 [REPORT.md](REPORT.md)。
- 图：`policy_replay.png`。

## 结果摘要（330 条流、1320 个阶段检查点）

| 方法 | 存储范围 | 遗忘范围（百分点） | 备注 |
|---|---|---:|---|
| sample (x,y) | 0.71～22.75 KB | 23.39 → **0.52** | 达到遗忘≤3pp 需 11.78 KB，≤1pp 需 20.63 KB |
| route_hold 锚点 | 0.73～23.25 KB | 24.20 → 26.04 | 平线，**从未达到 ≤3pp** |
| route 锚点 | 0.74～11.81 KB | 23.20 → 25.37 | 微弱且不随存储改善 |
| hold 锚点 | 0.73～11.69 KB | 29.21（与 no replay 相同） | 完全无效 |
| hybrid（锚点+极少量样本） | 3.62～26.09 KB | 19.24 → 9.55 | 同等字节下仍劣于纯 sample |
| none | 0 | 29.19 | 基线 |

- **核心负结果：Route+Hold policy replay 不能替代原始样本**。锚点存储从 0.7KB 加到 23KB，遗忘几乎没有变化；而纯样本 replay 在 5.7KB 时已达遗忘 8.87。
- **漂移洞察**：sample replay 的 ‖ΔRoute‖ 最大（4.40）却保留最好；策略蒸馏即使把 ‖ΔRoute‖ 压到 1.25（λ=50）也不改善遗忘。**遗忘的瓶颈不是控制策略漂移，而是状态动力学需要真实数据锚定。**
- λ 稳健性：蒸馏强度提高 100 倍，Route 漂移减半，遗忘 18.16→17.77（o0/rh192）——负结果对超参数稳健。
- none 与第十五轮 r=0 逐位一致 5/5。

## 运行（CPU）

```bash
cd flow_mvp_v16
for m in none sample route hold route_hold hybrid; do
  /mnt/d/betterNN/.venv/bin/python experiment.py --methods $m --orders o0 o1 o2 --out results --device cpu > logs/$m.log 2>&1 &
done
wait
/mnt/d/betterNN/.venv/bin/python analyze.py
/mnt/d/betterNN/.venv/bin/python plot.py
```

本轮共 1320 个阶段，是本仓库迄今最重的单轮（存储扫描条件数多），6 进程并行约 1 小时。

## 文件

- `experiment.py`：Flow-v2 + 五种存储方法 + 锚点收集与模块级蒸馏。
- `analyze.py`：全量复算、字节阈值插值、漂移汇总、λ 稳健性检查与报告。
- `plot.py`：`policy_replay.png`。
- `results/`：330 条流；`results_lambda/`：λ=10/50 稳健性检查。

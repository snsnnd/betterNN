# 第十七轮：可并行动力学可行性

Flow-v2 的 `h_t → G_t → h_{t+1}` 反馈串行深度为 O(T)。本轮在同一架构/参数量下只改接线，检验能力是否依赖严格时间递归，并验证可结合扫描的数学可行性。

- 设计见 [PLAN.md](PLAN.md)；完整结果见 [REPORT.md](REPORT.md)；扫描验证见 [scan_check.json](scan_check.json)。
- 图：`parallel_dynamics.png`。

## 四种接线

| 模型 | 底网 | Route 输入 | 反馈 | 可 scan |
|---|---|---|---|---|
| A flowv2 | tanh | [meta, pooled(h_t)] | 有 | ✗ |
| B linear | 线性 | [meta, pooled(h_t)] | 有 | ✗ |
| C preroute | 线性 | [meta, x_t, 0,0] | 无 | ✓ |
| D iter | 线性 | 两遍（预计算 + h⁰ 修正） | 无 | ✓ 两遍 |

## 结果摘要（120 条流）

| 模型 | r=12.5 最终% | r=12.5 遗忘 | 40 步最终% | 达到≤3pp 所需 replay |
|---|---:|---:|---:|---:|
| A flowv2 | **99.44** | **0.52** | **99.24** | ~6.7% |
| B linear | 92.86 | 5.54 | 76.91 | >12.5% |
| C preroute ✓ | 93.88 | 3.99 | 90.43 | >12.5% |
| D iter ✓ | 92.66 | 5.20 | 73.74 | >12.5% |

- **C（严格可 scan）在 CL 上仅落后 A 约 5.6pp 最终/3.5pp 遗忘，r=0 时几乎持平**；但 40 步外推落后 8.8pp，两遍迭代的 D 没有补回来（73.7）。
- **去掉 tanh（B）同样有代价**：replay 效率下降（final −6.6pp）且 40 步外推 −22pp，底层非线性是承重件。
- **扫描性质验证**：C 的顺序计算 vs Hillis–Steele 仿射扫描最大偏差 2.2e-9；D 第二遍 2.6e-9。朴素 CPU 扫描更慢属实现代价，本轮不做加速声明。
- **状态轨迹漂移**（r=0）：A 0.212、B 4.36、C 10.81、D 4.29——严格反馈模型的轨迹最稳定；但漂移幅度与遗忘不呈单调关系，不能单独作为遗忘指标。

## 结论

完全可并行化（C）能保住大部分持续学习能力，说明 **CL 任务本身不绝对依赖反馈闭环**；但长度外推明显依赖 `h→Route` 反馈与底层非线性。若目标是长序列，更现实的方向是**块并行**（如 16 步内递归、块间扫描），而不是完全去掉反馈。

## 运行（CPU）

```bash
cd flow_mvp_v17
FLOW_THREADS=4 /mnt/d/betterNN/.venv/bin/python scan_check.py
for m in flowv2 linear preroute iter; do
  /mnt/d/betterNN/.venv/bin/python experiment.py --models $m --orders o0 o1 o2 --ratios 0 .125 --out results --device cpu > logs/$m.log 2>&1 &
done
wait
/mnt/d/betterNN/.venv/bin/python analyze.py
/mnt/d/betterNN/.venv/bin/python plot.py
```

实测：`iter` 是最慢进程（两遍 rollout + 40 步外推评估），4 进程并行约 50 分钟。

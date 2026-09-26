# 第十轮：持续学习 benchmark 校准

为第十一轮 replay × 参数保护矩阵准备一个**每个任务独立可学**的干净基线。

- 设计、任务集修订与判定见 [PLAN.md](PLAN.md)。
- 实测结果见 [REPORT.md](REPORT.md)（由 `analyze.py` 生成）。
- 校准曲线与 CL 基线见 [benchmark.png](benchmark.png)。

## 新 benchmark A'B'C'D'

| 任务 | R=0 输出 | R=1 输出 | 说明 |
|---|---|---|---|
| A' | (A,B) | (B,A) | 与 legacy A 相同 |
| B' | (A,A) | (B,B) | 与 legacy B 相同 |
| C' | (A,B) | (A,A) | 新任务 |
| D' | (A,B) | (B,B) | 新任务 |

不存在互为规则逆映射的任务对。legacy A/B/C/D 保留在第八、九轮；旧校准（含病态互补任务 D）保存在 `results_legacy_calibration/` 作为证据。

## 结果摘要

- scratch 与 adapt 全部任务：测试 **99.75%～99.90%**，5/5 种子验证 ≥90%。
- 验证选预算：E90 = 48 轮，**E95 = 65 轮**（本 benchmark 固定用 65 轮/阶段）。
- 顺序 CL 基线（A'→B'→C'→D'、65 轮/阶段、5 种子）：acquisition **97.37 ± 3.24%**，最终平均 **80.80 ± 2.25%**，遗忘 **22.10 ± 5.86 个百分点**。

## 运行

轻量实验用 CPU 即可（N=256 时 CPU 比 GPU 快）。完整校准约 9 分钟，CL 基线约 3 分钟。

```bash
cd flow_mvp_v10
/mnt/d/betterNN/.venv/bin/python experiment.py --modes pretrain scratch adapt --epochs 150 --out results --device cpu
/mnt/d/betterNN/.venv/bin/python experiment.py --modes cl --epochs 65 --out results --device cpu
/mnt/d/betterNN/.venv/bin/python analyze.py --epochs 150
/mnt/d/betterNN/.venv/bin/python plot.py
```

## 文件

- `experiment.py`：A'B'C'D' 数据、scratch/adapt/CL 训练与检查点。
- `analyze.py`：全量状态复算、验证预算选择（写 `benchmark.json`）、报告生成。
- `plot.py`：`benchmark.png`。
- `results/`：校准与 CL 结果；`results_legacy_calibration/`：旧互补任务校准（失败证据）。
- `benchmark.json` / `REPORT.md` / `verification.txt`：预算判定、报告与核验。

# 第九轮：持续学习遗忘定位

回答第八轮判定之后的下一个问题：**A→B→C→D 顺序学习中，到底是谁在遗忘？**

- 设计、参数组与预先判定见 [PLAN.md](PLAN.md)。
- 实测结果见 [REPORT.md](REPORT.md)（由 `analyze.py` 生成）。

## 协议

- N=256 dense，任务与数据与第八轮完全一致。
- 先用全部参数学习 A 作为共同起点（每种子一个预训练检查点）。
- 再学 B/C/D，七组保护：`all`、`freeze_W`、`freeze_ctrl`、`freeze_readout`、`ctrl_only`、`readout_only`、`replay`。
- 5 个种子；另有 15 个独立适应对照（从同一预训练起点单独学 B/C/D）。
- replay 为等样本预算替换：每步 96 当前 + 32 旧任务样本。

## 运行

```bash
cd flow_mvp_v9
/mnt/d/betterNN/.venv/bin/python experiment.py      # 预训练 5 条 + 105 阶段 + 15 对照
/mnt/d/betterNN/.venv/bin/python analyze.py         # 重载核验 + REPORT.md
/mnt/d/betterNN/.venv/bin/python plot.py            # forgetting.png
```

筛选示例：`--arms all replay --seeds 11 --epochs 2 --out smoke --device cpu --skip-controls`。此时汇总与报告只写入 `smoke/`，不会覆盖主结果。

## 文件

- `experiment.py`：模型（与第八轮 dense 相同）、保护训练、replay 与对照。
- `analyze.py`：全量检查点复算、冻结参数核验、汇总与报告。
- `plot.py`：`forgetting.png`。
- `results/pretrain/`：每种子预训练检查点；`results/{arm}_{seed}_stage{task}.pt`：阶段检查点；`results/control_*.pt`：适应对照。
- `summary.json` / `all_metrics.json` / `paired_differences.json` / `controls_summary.json` / `REPORT.md` / `verification.txt`：汇总与报告。
- `forgetting.png`：各组最终成绩、遗忘与 A 任务保留曲线。

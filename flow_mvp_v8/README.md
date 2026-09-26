# 第八轮：Top-K 机制归因

回答第七轮遗留的首要问题：**Top-K 的提升来自可学习的选择，还是仅来自每任务随机支持集带来的稀疏隔离？**

- 设计、假说与判定规则见 [PLAN.md](PLAN.md)。
- 实测结果见 [REPORT.md](REPORT.md)（由 `analyze.py` 生成）。

## 环境（WSL, uv）

本目录使用项目根目录的 Linux 虚拟环境 `/mnt/d/betterNN/.venv`（由 uv 创建，Python 3.11，复用 uv 缓存中的 torch 2.14.0+cu130）：

```bash
uv venv /mnt/d/betterNN/.venv --python 3.11
uv pip install --python /mnt/d/betterNN/.venv/bin/python torch==2.14.0 numpy matplotlib
```

## 运行

```bash
cd flow_mvp_v8
/mnt/d/betterNN/.venv/bin/python experiment.py          # 15 条流，默认自动选择 GPU
/mnt/d/betterNN/.venv/bin/python parity_check.py        # 可选：CPU 下与第七轮存档逐位比对
/mnt/d/betterNN/.venv/bin/python analyze.py             # 重载核验 + summary + REPORT.md
/mnt/d/betterNN/.venv/bin/python plot.py                # attribution.png
```

筛选运行示例：`--seeds 11 --modes learned_topk --epochs 2 --out smoke --device cpu`。

## 三组定义

| 模式 | 选择器 | 活跃节点 | 说明 |
|---|---|---|---|
| dense | 不选择 | 256 | 全节点参照 |
| fixed_topk | 初始化后冻结 | 64 | 每任务随机支持集，制造“随机隔离” |
| learned_topk | 直通代理梯度 | 64 | 第七轮 Top-K 机制 |

同一 seed 下三组共享 W、B、mask、控制器、读出与选择器初始化；固定组与学习组前向数值相同，唯一差别是选择器是否接收梯度。所有流均 A→B→C→D、无回放、固定最后 epoch、不按测试选模型。

## 文件

- `experiment.py`：数据、模型、15 条流训练与保存。
- `parity_check.py`：与第七轮 CPU 存档的逐位一致性检查。
- `analyze.py`：检查点复算、配对差值、判定与报告生成。
- `plot.py`：`attribution.png`。
- `results/`：每流 JSON、最终检查点。
- `summary.json` / `all_metrics.json` / `paired_differences.json`：汇总。
- `verification.txt`：核验记录。

# 第十三轮：replay 保护机制定位

机制解释轮：梯度冲突 + 组件移植 + 缓冲大小扫描。

- 设计与发现摘要见 [PLAN.md](PLAN.md)；完整结果见 [REPORT.md](REPORT.md)。
- 图：`mechanism.png`。

## 运行

分析与移植直接复用第十一轮检查点（`../flow_mvp_v11/results`），API 顺序执行：

```bash
cd flow_mvp_v13
/mnt/d/betterNN/.venv/bin/python gradient_conflict.py     # conflict.json（截断 + 全 BPTT）
/mnt/d/betterNN/.venv/bin/python transplant.py            # transplant.json（8 组合）
for M in 1 2 4 8 16 32; do
  /mnt/d/betterNN/.venv/bin/python buffer_scan.py --orders o0 --archs flow --ratios .125 \
    --buffer-per-task $M --out results_buffer --device cpu > logs/m$M.log 2>&1 &
done
wait
/mnt/d/betterNN/.venv/bin/python analyze.py
/mnt/d/betterNN/.venv/bin/python plot.py
```

## 关键结果

- **Write 从未被训练**：5 步截断下梯度恒为 0（输入在 t=0/1，梯度窗口只有最后 5 步）。
- **旧策略在 Controller + W**：换回 ctrl 恢复 A 到 91.39%，ctrl+W 到 99.75%，readout 无影响。
- **replay 降低冲突**：hold 的旧/新任务梯度 cosine 由负转正；α* 中位数 ≤2.83%。
- **缓冲阈值**：M=8 时遗忘 6.80，M=32 时 0.73 个百分点。

## 文件

- `mechanism.py`：数据/模型/评估（与第十二轮 flow 路径一致）。
- `gradient_conflict.py` / `transplant.py` / `buffer_scan.py`：三类分析。
- `conflict.json` / `transplant.json` / `results_buffer/`：原始结果。
- `summary.json` / `REPORT.md` / `verification.txt` / `mechanism.png`：汇总。

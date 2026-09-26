# 第十四轮：Write 因果归因

判定 Write 门是否必要：frozen（随机固定）vs const1 / const05（常数写入）vs full_bptt / aux（真正训练 Write）。

- 设计见 [PLAN.md](PLAN.md)；完整结果见 [REPORT.md](REPORT.md)。
- 图：`write_attribution.png`。

## 结果摘要（50 条流）

| Write 模式 | r=0 最终% | r=0 遗忘 | r=12.5 最终% | r=12.5 遗忘 | ‖ΔWrite‖ |
|---|---:|---:|---:|---:|---:|
| frozen（现状） | 80.80 | 22.10 | 99.15 | 0.73 | 0.000 |
| const1 | 81.61 | 22.52 | 99.04 | 0.91 | 0.000 |
| const05 | 81.62 | 22.64 | 98.29 | 1.97 | 0.000 |
| full_bptt（真训练） | 81.56 | 22.54 | 98.28 | 1.86 | 1.869 |
| aux（局部监督） | 80.85 | 25.51 | 99.25 | 0.96 | 4.628 |

- 常数写入与随机固定 Write 无显著差异（r=0 差值 +0.8 个百分点、配对 3/5；r=12.5 差 −0.1～−0.9）。
- 真正训练 Write（full_bptt，ΔWrite≈1.9）或局部监督（aux，ΔWrite≈4.6～6.3）也没有稳定收益。
- **结论：本 benchmark 上随机 task-conditioned 写入没有可测贡献，Write 门可从架构中删除，输入注入可直接用常数。**
- frozen 与第十一轮逐位一致 10/10。

## 运行（CPU，5 模式可并行）

```bash
cd flow_mvp_v14
for w in frozen const1 const05 full_bptt aux; do
  /mnt/d/betterNN/.venv/bin/python experiment.py --write-modes $w --ratios 0 .125 --out results --device cpu > logs/$w.log 2>&1 &
done
wait
/mnt/d/betterNN/.venv/bin/python analyze.py
/mnt/d/betterNN/.venv/bin/python plot.py
```

## 文件

- `experiment.py`：5 种写入模式、replay 训练与阶段检查点。
- `analyze.py`：全量复算、配对差值与判定。
- `plot.py`：`write_attribution.png`。
- `results/`：50 条流 JSON 与 200 个阶段检查点。

# 第二十三轮：Credit-Stress Benchmark（长程 core 信用压力任务）

- 设计见 [PLAN.md](PLAN.md)（含运行期诊断 Addendum）；结果见 [REPORT.md](REPORT.md)（`analyze.py` 自动生成）；救援诊断见 [probe_rescue.py](probe_rescue.py)。
- 问题：V22 在 A'B'C'D' 上没找到 core 长程信用需求。V23 **不改模型、只造任务**（Chain-select：A/c1/distractor/B/c2 全部瞬态脉冲 + 长间隔 + 最后读出），扫描 `T∈{20,40,80,160}` × core 窗口 `K∈{5,10,20,Full}`（B 恒 Full），回答是否存在"5 步明显失败、Full 明显成功"的硬案例。

## 主要结果

- **没有找到长程硬案例**：T≥40 时 K=5 的可达性（best-val≥0.9）≥ Full——T=40 xor：K5 5/5 vs Full 3/5；T=80 xor：K5 4/5 vs Full 3/5；T=160：K5 2/3 vs Full 2/3。
- **唯一干净的窗口效应是短 T 的时序对齐**：xor T=20 时 K=5 可达 0/5（y0 卡在 ~0.75）、K=10 5/5、Full 4/5；K=5 末窗 [15..19] 只含 c2@15、缺 B@12，K=10 窗口覆盖 B。
- **主失败模式是"软选择平台/周期振荡"**：y0≈0.75–0.94；Full 同样中招（T=40/80 可达仅 3/5；T=160 出现 val 1.0 → final 0.38 的崩溃），所以 final 的 K 差异不能归因于 credit horizon。
- **救援诊断**：T=20 xor 的 K=5 失败流续训 Full 100 epochs 无一条被救回（0.750→0.733）——是坏盆吸引子，不是表示能力/信用不足。
- **sustained 对照**（c1/c2 持续可见）：Full 5/5 可达，K5 仅 2/5（xor）/4/5（xorsw），与 transient 方向不一致 → 瓶颈不是"命令是否可见"。
- 结论：**core 不需要长程信用（强化 V22）**；V24（eligibility/local credit）暂不应启动；Flow-v3 继续只解决 B 的跨窗口信用，且"更长 core 信用"并非免费（Full 稳定性更差）。

## 运行

```bash
# 主网格：transient × {xor,xorsw} × T{20,40,80} × K{5,10,20,full} × 5 seeds
FLOW_THREADS=1 python experiment.py --variants transient --tasks xor xorsw --Ts 20 40 80 \
    --Ks 5 10 20 full --seeds 11 22 33 44 55 --epochs 500 --jobs 6 --out results/main
# sustained 对照（T=80）
FLOW_THREADS=1 python experiment.py --variants sustained --tasks xor xorsw --Ts 80 \
    --Ks 5 20 full --seeds 11 22 33 44 55 --epochs 500 --jobs 3 --out results/control
# T=160 缩放（xor、3 seeds、K{5,20,full}）
FLOW_THREADS=1 python experiment.py --variants transient --tasks xor --Ts 160 \
    --Ks 5 20 full --seeds 11 22 33 --epochs 500 --jobs 3 --out results/scale
# 救援诊断（K=5 → Full 微调）
FLOW_THREADS=1 python probe_rescue.py 11 22 33 44 55
# 汇总
python analyze.py && python plot.py     # REPORT.md / credit_stress.png
```

`--Ks` 中 `full` 表示 core 完整 20/T 步信用；`K<Full` 用 V22 双通道（core 截断 / B+readout 完整）。T=160 单流约 20–45 min、RSS≈1.1GB，注意并行度。

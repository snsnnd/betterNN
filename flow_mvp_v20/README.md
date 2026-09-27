# 第二十轮：B 输入拓扑 × 底网动力学

- 设计见 [PLAN.md](PLAN.md)；结果见 [REPORT.md](REPORT.md)（由 `analyze.py` 自动生成）。
- 三阶段：**A** 主族 topology（single-random / single-central / multi8 / multi16 / distributed）、
  **B** shared-pool overlap（α=0/0.5/1）、**C** `s_W=‖W₀‖₂∈{0.7,0.9,1.1}` × {single-random, distributed}。
- 统一协议：A'B'C'D'、5 seeds、65 轮/阶段、orders o0/o1/o2、r∈{0,12.5%}；每通道输入能量 `‖B_c‖₂=5.6`。
- 指标：单任务可学性、CL（acquisition/final/forgetting）、解析动力学（ρ_eff、γ₄/γ₈）、state lifetime（τ½、G_max）、
  可达性（正式任务 PCA 有效秩 + 非训练 probe）、tanh 饱和。

## 运行

```bash
# 训练（每 phase 一个进程，内部 Pool 并行；A 约 40min/12jobs，B/C 更短）
FLOW_THREADS=1 python experiment.py --phase A --jobs 12 --out results/A
FLOW_THREADS=1 python experiment.py --phase B --jobs 6  --out results/B
FLOW_THREADS=1 python experiment.py --phase C --jobs 8  --out results/C
# 汇总：完整性/回归核验 + 单任务指标（缓存 results/*/metrics_singles.json）+ H1/H2/H3 + REPORT.md + h3.json
FLOW_THREADS=8 python analyze.py
python plot.py     # input_topology.png / overlap_gradient.png / sW_interaction.png
```

单配置 smoke（2 轮）：

```bash
python experiment.py --phase A --topos single-random --seeds 11 --orders o0 --ratios 0 .125 --epochs 2 --out smoke
python metrics.py smoke/single_single-random_t0_11.pt --task 0    # 指标单测
```

产物：`results/{A,B,C}/*.json`、`*_stage{i}.pt`、`metrics_singles.json`；根目录 REPORT.md 与三张图。

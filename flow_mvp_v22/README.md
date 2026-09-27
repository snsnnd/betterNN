# 第二十二轮：Long-range Credit Attribution

- 设计见 [PLAN.md](PLAN.md)；结果见 [REPORT.md](REPORT.md)（`analyze.py` 自动生成）；回归记录 [verification.txt](verification.txt)。
- 问题：V21 发现 **hybrid（B 全 BPTT / core 5 步截断）→ Full BPTT** 能把 r=12.5% 遗忘从 6.63pp 降到 1.83pp。V22 只做一件事：把完整 20 步信用逐组发给 `{W, Route, Hold}` 的 2³ 子集（B 恒 Full），回答"到底谁补上了长程信用"。
- 协议：每个 step 同一参数快照双通道（截断 period=5 / 完整 20 步），按组拼装梯度；双 optimizer、分别 clip，其余与 V21 hybrid 一致。`B` 臂与 V21A hybrid **逐位一致**（max|Δmatrix|=0）。

## 主要结果

- **协议核验**：`B` 臂与 V21A hybrid **逐位一致**（max|Δmatrix|=0，5 流）；首 batch 上 V22 完整通道梯度与 V21D 的 backward 逐位相同（loss bitwise equal、grad maxdiff=0）；`All` 与 V21D 矩阵差 0.038（joint vs separate clip 在 1040 步轨迹上的放大，clip 实测会生效 max norm≈1.9–2.5），但遗忘均值 1.95 vs 1.83（o0/5 seeds）在结论层面一致。
- **主结果（r=12.5%，3 顺序 × 10 seeds）**：G=F(B)−F(All)=**0.17pp**；ME_W/R/H = −0.14/+0.05/+0.36pp（同向 5/10、4/10、5/10）。没有任何子集的长程信用能稳定降低遗忘：H1/H3/H4/H5 未通过，H2 以 |ME_W|≤1pp 通过（G≈0 时 closure 不可解释）。
- **V21 gap 重审**：V21 的表面 gap（6.63 3 顺序 vs 1.83 仅 o0 = 4.80pp）主要来自聚合错配；匹配同 5 seeds/3 顺序 gap = **1.11pp**，10 seeds/3 顺序 = **0.17pp**；分顺序 B−All = +1.08（o0，5/10）/ +1.39（o1，6/10）/ −1.94（o2，4/10）。
- **r=0 负控**：所有 arm 24–25pp，|ME|≤0.4pp——信用窗口只在 replay 下才有（微弱）作用。
- **credit audit（V21 检查点）**：训练后 5 步梯度相对 20 步的方向 cos：W≈0.33（模长 6–10%）、hold≈0.50–0.65、route≈0.74–0.84；readout=1、B=0。窗口损失真实存在，但不转化为功能差异。
- **含义**：B 的完整信用是必要杠杆（V21），而 core 的 5 步窗口在 A'B'C'D' 上足够；Flow-v3 的 local/online 长程信用需要先在能真正暴露窗口缺陷的协议（更长序列/多事件/部分可观测）上复现窗口效应。

## 运行

```bash
# Phase 0：B 臂逐位回归 + All 臂对 V21D（结果在 results/reg）
FLOW_THREADS=1 python experiment.py --arms B All --seeds 11 --epochs 65 --orders o0 --ratios .125 --out results/reg --no-single
# Phase 0：V21 检查点上的逐组信用审计（results/credit_audit.json）
FLOW_THREADS=2 python experiment.py --credit-audit --seeds 11 22 33 44 55 --out results
# 主运行（两批；8 arms × 10 seeds × o0/o1/o2 × r=12.5% + r=0 o0 + singles）
FLOW_THREADS=1 python experiment.py --seeds 11 22 33 44 55 66 77 88 99 110 --orders o0 --ratios .125 --jobs 28 --out results/A
FLOW_THREADS=1 python experiment.py --seeds 11 22 33 44 55 66 77 88 99 110 --orders o1 o2 --ratios .125 --no-ckpt --no-single --jobs 5 --out results/A
FLOW_THREADS=1 python experiment.py --seeds 11 22 33 44 55 66 77 88 99 110 --orders o0 --ratios 0 --no-ckpt --no-single --jobs 5 --out results/A
python analyze.py && python plot.py     # REPORT.md / attribution.png / credit_audit.png
python verify_protocol.py               # 协议核验（追加到 verification.txt）
```

`--credit-audit` 只读 V21 检查点；训练协议除逐组信用窗口外与 V21 hybrid 完全一致。

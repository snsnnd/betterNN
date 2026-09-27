# 第二十一轮：Adaptive Input Decoupling

- 设计见 [PLAN.md](PLAN.md)；结果见 [REPORT.md](REPORT.md)（`analyze.py` 自动生成）；回归记录 [verification.txt](verification.txt)。
- 问题：B 能否自己学会把两个输入源写进不同的状态子空间；显式解耦正则是否有额外价值；`B overlap → state overlap → gradient conflict → forgetting` 链是否成立。
- 关键协议：**hybrid 信用分配**（core 5 步截断 / B 全 BPTT；同一参数快照、双 optimizer、core/B 分别 clip）；写入域仅 role0∪role1，`‖B_eff‖₂=5.6`；overlap penalty 用归一化 `O_B`。

## 主要结果

- **fixed 回归**：V21 fixed 臂与 V20 检查点逐流矩阵差 = 0（12 条），证明 fixed 协议未变。
- **H1（复现）通过（主条件）**：fixed-disjoint vs fixed-overlap，r=12.5% Δforgetting = −3.39pp、4/5 负向；r=0 仅 −1.84pp（方向复核未过，与 V20 一致）。
- **H2a 未通过**：learnable (λ=0) 自发把 O_B 从 1.0 降到 init 的 0.83–0.91（0/5 达 ≤0.75）——有自发解耦，但幅度不足以达标。
- **H2b**：overlap penalty 强烈解耦（O_B→0.12/0.001，ΔO_B=−0.73/−0.86，两档方向一致 → 通过）；orth 在已近似正交的初始化上几乎无作用。**但 hybrid 主实验中解耦并未换来显著遗忘下降**（Δforget ≤0.6pp）。
- **H3（机制链）部分**：ρ(O_h,C∇)=0.59 通过；ρ(O_B,F)=0.41 边际；ρ(O_B,O_h)=0.21、ρ(C∇,F)=0.09 未达标。
- **V21D（Full BPTT 敏感性，40 流）**：r=12.5% 时 fixed-overlap 遗忘 4.66pp，learnable / learnable+overlap0.1 降到 1.83/1.86（达到 fixed-disjoint 1.95 水平）；r=0 各臂 ≈24pp 无差异。即“B 有完整长程信用时，自适应解耦能消除 overlap 代价；hybrid 下不能”。
- **trunc 负控**：`|∇B| = 0`（5 seeds，credit check），与 v13 Write 同源。

## 运行

```bash
# 协议提交后（无结果）：
FLOW_THREADS=8 python experiment.py --credit-check --seeds 11 22 33 44 55   # Phase 0
# V21A/B/C/D + fixed 回归
FLOW_THREADS=1 python experiment.py --arms learnable learnable+orth0.1 learnable+overlap0.1 --jobs 8 --out results/A
FLOW_THREADS=1 python experiment.py --arms learnable+orth1 learnable+overlap1 --jobs 9 --out results/B
FLOW_THREADS=1 python experiment.py --arms learnable-random --jobs 4 --out results/C
FLOW_THREADS=1 python experiment.py --arms fixed-overlap fixed-disjoint learnable learnable+overlap0.1 \
    --seeds 11 22 33 44 55 --orders o0 --ratios 0 .125 --jobs 4 --out results/D --b-grad full
FLOW_THREADS=1 python experiment.py --arms fixed-overlap fixed-disjoint --seeds 11 --jobs 2 --out results/fixed_reg
FLOW_THREADS=8 python analyze.py && python plot.py     # REPORT.md / decoupling.png / fullbptt_sensitivity.png
```

`--b-grad hybrid|full|trunc` 控制 B 的信用分配；hybrid 为预注册主协议。

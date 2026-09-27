# 第二十一轮：Adaptive Input Decoupling（B 自发/正则解耦）

协议：hybrid 信用分配（core 5 步截断 / B 全 BPTT，双 optimizer、分别 clip）；写入域 role0∪role1；`‖B_eff‖₂=5.6`；A'B'C'D'、5 seeds、o0/o1/o2、r∈{0,12.5%}。

## 1. fixed 回归核验（v21 vs V20 检查点）

- 逐流矩阵最大差：0.00（12 条配对）；应为 0，证明 fixed 臂协议未变。

## 2. CL 汇总（合并 3 顺序，均值）

### r=0%

| arm | acquisition | final_mean | forgetting (pp) | O_B init→final | D_B | O_h(t8) | C∇ |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed-disjoint | 0.981 | 0.757 | 29.81 | 0.000→0.000 | 0.000 | — | — |
| fixed-overlap | 0.990 | 0.753 | 31.64 | 1.000→1.000 | 0.000 | — | — |
| learnable | 0.989 | 0.758 | 30.78 | 1.000→0.890 | 0.293 | 0.494 | 0.264 |
| learnable+orth0.1 | 0.989 | 0.757 | 30.95 | 1.000→0.883 | 0.298 | 0.494 | 0.239 |
| learnable+orth1 | 0.989 | 0.758 | 30.75 | 1.000→0.883 | 0.296 | 0.495 | 0.291 |
| learnable+overlap0.1 | 0.986 | 0.761 | 30.01 | 1.000→0.144 | 0.641 | 0.393 | 0.228 |
| learnable+overlap1 | 0.982 | 0.759 | 29.64 | 1.000→0.001 | 0.915 | 0.359 | 0.289 |
| learnable-random | 0.981 | 0.746 | 31.39 | 0.601→0.593 | 0.236 | 0.360 | 0.206 |

### r=12.5%

| arm | acquisition | final_mean | forgetting (pp) | O_B init→final | D_B | O_h(t8) | C∇ |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed-disjoint | 0.977 | 0.943 | 5.65 | 0.000→0.000 | 0.000 | — | — |
| fixed-overlap | 0.980 | 0.915 | 9.04 | 1.000→1.000 | 0.000 | — | — |
| learnable | 0.984 | 0.936 | 6.63 | 1.000→0.837 | 0.301 | 0.501 | 0.206 |
| learnable+orth0.1 | 0.983 | 0.939 | 6.01 | 1.000→0.826 | 0.303 | 0.505 | 0.174 |
| learnable+orth1 | 0.983 | 0.928 | 7.30 | 1.000→0.830 | 0.300 | 0.511 | 0.191 |
| learnable+overlap0.1 | 0.984 | 0.937 | 6.71 | 1.000→0.124 | 0.643 | 0.400 | 0.234 |
| learnable+overlap1 | 0.985 | 0.939 | 6.17 | 1.000→0.001 | 0.911 | 0.347 | 0.236 |
| learnable-random | 0.987 | 0.953 | 4.75 | 0.601→0.583 | 0.252 | 0.355 | 0.213 |

## 3. 预注册判定

### H1（fixed-disjoint > fixed-overlap）

- r=12.5%（主条件）：Δforgetting = -3.39pp，负向 4/5；门槛 −2pp → 通过
- r=0%（方向复核）：Δforgetting = -1.84pp，负向 3/5；门槛 −2pp → 未通过

### H2a（learnable λ=0 自发解耦）

- O_B^final/O_B^init 按 seed：[0.857, 0.828, 0.885, 0.909, 0.84]；≤0.75 的 seeds 0/5 → 未通过

### H2b（penalty 相对 λ=0 的增量）

- orth：方向一致 True，判定 False
  - learnable+orth0.1: ΔO_B=-0.009, Δforget=-0.62pp, ΔC∇=-0.029
  - learnable+orth1: ΔO_B=-0.007, Δforget=+0.67pp, ΔC∇=+0.006
- overlap：方向一致 True，判定 True
  - learnable+overlap0.1: ΔO_B=-0.730, Δforget=+0.08pp, ΔC∇=-0.004
  - learnable+overlap1: ΔO_B=-0.863, Δforget=-0.46pp, ΔC∇=+0.027

### H3（机制链 Spearman，arm×seed）

- n=40；ρ(O_B,O_h)=0.21、ρ(O_h,C∇)=0.59、ρ(C∇,F)=0.09、ρ(O_B,F)=0.41；门槛 0.4（r=0）

## 4. 边界

- mediation 为 exploratory（5 seeds）；V21D（Full BPTT 40 流）与 V21-trunc 负控单独报告。
- fixed 臂直接复用 V20（fixed_reg 已核验）；O_h 为 counterfactual 输入响应子空间主角均值（r=8, t=8）。

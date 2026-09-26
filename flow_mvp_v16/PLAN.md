# 第十六轮：Sample replay vs Route/Hold Policy replay

## 问题

Flow-v2（W+Route+Hold+Readout）已冻结。第 13/14 轮表明旧任务策略分布在 Route+Hold+W。本轮只改"旧知识怎么保存"，以**存储字节数**为横轴回答：

> **只保存 Route/Hold 的旧策略，能否用比原始样本更少的字节达到同样的遗忘水平？**

## 存储定义与单条成本（float32）

| 方法 | 保存内容 | 每任务字节/条 |
|---|---|---|
| `sample` | (x, meta, y)：20×2+20×7+2=182 floats | 728 B |
| `route` | Controller 输入 q=[meta,pooled(h)]（11）+ Route logits（16） | 108 B |
| `hold` | meta（7）+ Hold logits（4） | 44 B |
| `route_hold` | q + Route logits + Hold logits = 31 floats | 124 B |
| `hybrid` | route_hold anchors + 极少量原始样本 | 124 A + 728 S |

Policy replay 用**模块级函数蒸馏**：保存旧的 (输入, sigmoid 前 logits) 锚点，训练新任务时加
`L = L_new + λ·MSE(Route_new(q_anchor), r_old) + λ·MSE(Hold_new(meta_anchor), u_old)`。
保存 logits 避免 sigmoid 饱和影响蒸馏。

## 固定项与扫描

- Flow-v2、N=256、A'B'C'D'、65 轮/阶段、每步 128 样本、每步旧信息带宽固定 16 条（12.5%），3 顺序 × 5 种子。
- 存储预算（每任务）：
  - `sample` M∈{1,2,4,8,16,32} → 0.7～23.3 KB
  - `route_hold` A∈{6,12,24,48,96,192} → 0.7～23.8 KB
  - `route` A∈{7,28,112} → 0.76～12.1 KB
  - `hold` A∈{17,68,272} → 0.75～12.0 KB
  - `hybrid`：(24,1)、(96,2)、(192,4) → 3.7～26.7 KB
  - `none`：0 B
- 同时记录漂移：`‖ΔRoute‖、‖ΔHold‖、‖ΔW‖、‖ΔReadout‖`（最终 vs stage0）。

## 判定

- 核心指标：**Forgetting vs Memory Bytes**；比较各方法把遗忘压到 ≤3pp / ≤1pp 所需的字节数。
- 若 policy replay 用显著更少字节达到同样的遗忘 → 支持"控制策略可作为压缩的长期记忆表示"。
- 若 policy replay 只能到中等保留、加极少量样本后跃升 → 存在"策略记忆 + 底层动力记忆"两部分。
- `none` 应与第十五轮 r=0 逐位一致；`sample` 曲线应与第十三轮缓冲扫描对齐。

## 边界

- Policy 蒸馏保存的是"输入→旧策略输出"的函数锚点，不存原始输入；锚点为旧任务训练分布上的采样。
- λ=0.5；每步带宽与样本法一致（16 条），但蒸馏算子在控制器模块上，计算量不同。

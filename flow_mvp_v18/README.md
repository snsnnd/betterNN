# 第十八轮：非线性 Flow 的 block-affine 可压缩性（诊断）

不训练、不引入新架构：在第十五轮训练好的 Flow-v2 上，测"块仿射 + 迭代重线性化"能恢复多少精度、需要多大有效串行深度。

- 设计与原委见 [PLAN.md](PLAN.md)；完整结果见 [REPORT.md](REPORT.md)。
- 图：[compressibility.png](compressibility.png)。
- 上一稿"块 Jacobi 多遍传播"已排除，保留在 [block_jacobi.py](block_jacobi.py) 作为失败记录。

## 方法

把 T 等分为 B 块；每遍在入口估计 p 处计算块函数 `F_i(p_i)` 与 Jacobian 方向导数（`torch.func.jvp`），按仿射组合 `ĥ_{i+1}=F_i(p_i)+J_i(ĥ_i−p_i)` 传播，再把 `ĥ` 作为下一遍的线性化点（K=0..3）。初始估计：`zero` 或第十七轮训练好的 `preroute` 模型轨迹。有效深度按 (K+1)·L 计。

## 关键结果（5 种子 × 4 任务 × 256 样本；Flow 参考 acc40 99.82%、PreRoute 模型 86.37%）

| 初始估计 | L | K（重线性化次数） | 有效深度 | acc40% | E_h |
|---|---:|---:|---:|---:|---:|
| preroute | 1 | 2 | 3 | 97.34 | 0.071 |
| preroute | 2 | 2 | 6 | 98.14 | 0.018 |
| preroute | 4 | 2 | 12 | **99.26** | 0.0045 |
| preroute | 5 | 2 | 15 | **99.57** | 0.0025 |
| preroute | 2 | 3 | 8 | **99.79** | 0.0005 |
| zero | 10 | 2 | 30 | 98.75 | 0.004 |
| zero | 4 | 3 | 16 | 94.63 | 0.003 |

- **Oracle 自检精确为 0**：在真实入口处线性化再组合恒等于原轨迹（这是恒等式，不构成压缩损失；损失来自近似入口）。
- **有并行预测器时压缩性很好**：PreRoute 轨迹 + 2 次重线性化，L=4～5 就能把 40 步恢复到 ≥99.26%（Flow 99.82%），有效深度 12～15。
- **无预测器（zero）也收敛但慢**：达到 99% 需要深度 20～40；预测器显著减少所需遍数。
- **误差放大**：块 Jacobian σ_max≈10.25，扰动每块放大约 10 倍；但迭代重线性化在实践中仍收敛（E_h 3 遍后 ~10⁻³～10⁻⁴）。
- 代价说明：每个重线性化遍需要块 Jacobian 的作用（本测量用 batched JVP；真实并行实现需要显式 M_i 或高效 scan），因此这轮证明的是**深度可压缩**，不是计算量已降低。

## 运行（CPU，约 3 分钟）

```bash
cd flow_mvp_v18
for s in 11 22 33 44 55; do
  /mnt/d/betterNN/.venv/bin/python block_affine.py --seeds $s --out ba_seed$s.json --sigma-diagnostic > logs/ba$s.log 2>&1 &
done
wait
/mnt/d/betterNN/.venv/bin/python analyze.py
/mnt/d/betterNN/.venv/bin/python plot.py
```

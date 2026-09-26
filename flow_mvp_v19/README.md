# 第十九轮：并行求解器可行性（Phase 0B 结构扫描 + Phase 0C GPU 基准）

- 设计见 [PLAN.md](PLAN.md)；Phase 0B 结果见 [REPORT.md](REPORT.md)；验证记录见 [verification.txt](verification.txt)。
- 已完成：**Phase 0B**（解析块仿射结构 + scan tree 秩/截断 + L=4 求解侧测）与 **Phase 0C**（GPU batch 化 solver 实现与计时：serial / GS-JVP / structured scan / structured seq）。
- 未完成：Phase 1（boundary predictor）、Phase 2（冻结 solver 评估）、Phase 3（joint training）。

## Phase 0B 结论

1. 解析分解 `J_t = S_t + C_t`（C 来自 `pooled(h)→Route`，rank ≤4）与 autograd 一致到 float32 噪声底（单步 2e-8、块 1.7e-7），S 保持 8/16 角色块支撑。
2. 块修正秩均值 6.7～8.6（L=4，界 16）、8.6（L=10 40 步，界 40）；scan tree 秩逐层饱和在 ~12.3，不向 256 膨胀。
3. 截断到 r=16 的相对误差 ~9e-6，r=8 约 8e-4，r=4 约 2e-2 —— r=16 已接近无损。
4. L=4 求解侧测：20 步下任意初值 K=2~3 遍收敛到 ~1e-8；40 步下 pre/noise 多数 job 收敛到 <1e-5，但少数 (seed44 task2/3 等) 从扰动中心出发会震荡发散。
5. 回归：v18 compose 与 v19 求解链在 32 个 (配置,K) 上 acc 完全一致、max|ΔEh| 8.7e-5；与 v18 存储 JSON（batch 256）max|Δacc| = 0。

## Phase 0C 结论（RTX 4060 Laptop、fp32、eager PyTorch）

数据：`results/solver_bench.json`（PreRoute 中心）、`results/solver_bench_zero.json`（zero 中心）；图 `solver_pareto.png`。T∈{40,128,512}、batch∈{1,8,32}、L=4、K∈{0,1}。

1. **精度一致性**：同一中心下 structured 与 GS-JVP 的 acc 完全相同（如 T=40 zero 中心 K=0 均 0.586、K=1 均 0.391），但与 serial（1.000）差距大——问题在初值/迭代，不在算子等价性。
2. **latency：所有并行变体在 eager 实现下都不占优。** serial 本身是 launch-bound（T=512: ~260–390ms）；GS-JVP 比 serial 慢 6～13×；structured scan 慢 2～35× 且在 B≥32 时溢出/报错。
3. **唯一比 serial 快的变体是 structured seq（逐块顺序 apply，无 scan）**：T=128 b=1 K=0 为 32.9ms vs serial 66.9ms、T=512 b=1 K=0 为 72.2ms vs 308ms；但 K≥1 时因在坏中心上重线性化而数值爆炸（nan/ERR），batch 32 时也无优势。
4. **精确 affine scan 有数值硬限制**：扫描需要跨块 Jacobian 乘积 `M_B···M_1`，即使 zero 中心下也会以 ~ρ^B 增长并溢出 fp32（T=128 B=32 起持续 OverflowError）。要扫描必须引入缩放/归一化或改变算子表示。
5. **PreRoute 不能做长序列初值**：其状态 max|h| 在 T=40/128/512 分别为 1.3e3 / 2.1e10 / NaN（无 tanh 的线性模型在长序列发散）。Phase 1 必须按目标长度训练 predictor。
6. **判定**：按 PLAN 的 0C 门槛（latency ≤ serial、显存不炸、acc 达标），当前原型**未通过**，因此不进入 Phase 3；Phase 1（冻结 Flow 的 predictor 蒸馏）仍然独立且必要。下一步优先级：更好的 predictor（Phase 1）→ 若重开 solver 优化，需要融合/编译（CUDA Graph/torch.compile）与稳定化扫描。

## 运行

```bash
FLOW_THREADS=1 python experiment.py --jobs 12 --out results/raw      # Phase 0B 全量结构扫描，约 7 分钟
FLOW_THREADS=8 python analyze.py                                     # 聚合 + 回归 + 写 REPORT.md
python plot.py                                                       # 三张图
FLOW_THREADS=8 python solver_bench.py --check                        # solver 与 CPU 参考一致性
FLOW_THREADS=8 python solver_bench.py --steps 40 128 512 --batch 1 8 32 --ks 0 1 --init zero \
    --out results/solver_bench_zero.json                             # Phase 0C 基准（约 10 分钟/组）
```

`analyze.py` 会把回归结果缓存到 `results/regression.json`（加 `--refresh-reg` 强制重算）。

数据：`results/raw/{seed}_{task}_{steps}.json`（40 个 job）、`results/summary.json`、`results/regression.json`、`results/solver_bench*.json`。

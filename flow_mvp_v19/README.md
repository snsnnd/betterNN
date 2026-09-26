# 第十九轮：并行求解器可行性（Phase 0B：算子结构扫描）

- 设计见 [PLAN.md](PLAN.md)；完整结果见 [REPORT.md](REPORT.md)；验证记录见 [verification.txt](verification.txt)。
- 本目录当前只覆盖 **Phase 0B（无训练诊断）**：解析 Flow-v2 块 Jacobian 的 `S + UVᵀ` 结构、验证 `rank(J−S) ≤ 4L`、测量 scan tree 秩增长与截断误差，并跑 L=4 的求解侧测（中心初值 × K 遍）。
- Phase 0A/0C（GPU 计时与 solver 接线对比）与 Phase 1/2/3（predictor、冻结 solver、joint training）尚未实现。

## 主要结论

1. 解析分解 `J_t = S_t + C_t`（C 来自 `pooled(h)→Route`，rank ≤4）与 autograd 一致到 float32 噪声底（单步 2e-8、块 1.7e-7），S 保持 8/16 角色块支撑。
2. 块修正秩均值 6.7～8.6（L=4，界 16）、8.6（L=10 40 步，界 40）；scan tree 秩逐层饱和在 ~12.3，不向 256 膨胀。
3. 截断到 r=16 的相对误差 ~9e-6，r=8 约 8e-4，r=4 约 2e-2 —— r=16 已接近无损。
4. L=4 求解侧测：20 步下任意初值 K=2~3 遍收敛到 ~1e-8；40 步下 pre/noise 多数 job 收敛到 <1e-5，但少数 (seed44 task2/3 等) 从扰动中心出发会震荡发散——Phase 2 需要阻尼或更好初值。
5. 回归：v18 compose 与 v19 求解链在 32 个 (配置,K) 上 acc 完全一致、max|ΔEh| 8.7e-5；与 v18 存储 JSON（batch 256）max|Δacc| = 0。

## 运行

```bash
FLOW_THREADS=1 python experiment.py --jobs 12 --out results/raw   # 全量结构扫描，约 7 分钟（12 进程）
FLOW_THREADS=8 python analyze.py                                  # 聚合 + 回归 + 写 REPORT.md
python plot.py                                                    # operator_structure.png / solver_convergence.png
```

`analyze.py` 会把回归结果缓存到 `results/regression.json`（加 `--refresh-reg` 强制重算）。

数据：`results/raw/{seed}_{task}_{steps}.json`（40 个 job）、`results/summary.json`、`results/regression.json`。

# 第二十五轮（V24D-Phase 0）：Temporal Strategy 可发现性审计

- 设计见 [PLAN.md](PLAN.md)（预注册），结果见 [REPORT.md](REPORT.md)（`analyze.py` 自动生成）。
- 问题：延迟任务上 fixed kernel 已知有更优解（65ep），但 V24C adaptive selector 未发现它。**失败来自表示限制，还是优化/探索限制？** 本轮只审计，不跑连续 λ 主实验。
- 矩阵（T=80、task0、B fixed、core K=5、scheduler hybrid full、3 seeds）：

| arm | 参数化 | 初始化 | epochs |
|---|---|---|---|
| A1 | bank | single | 65（V24C 回归锚点） |
| A2/A3 | bank | burst5 / decay-slow | 65 |
| A4 | bank | single | 500 |
| B1/B2 | shared λ（单标量） | λ0=0.05 / 0.6 | 65 |
| B3 | shared λ | λ0=0.05 | 500 |
| F1/F2/F3 | fixed single/burst5/decay-slow | — | 500 |

## 主要结果

- **回归锚点**：A1 与 V24 `bank_delay_T80_t0_*` 同 seed **3/3 逐位一致**（max|Δfinal|=0）。
- **关键：预算是混淆**。65ep 上 best fixed 领先 single +5.8/+29.6/+16.2pp（seeds 11/22/33）；到 500ep 收缩到 ±4pp 以内（F1 0.965±0.017 / F2 0.979±0.001 / F3 0.975±0.011），且 A4/B3 与 fixed single 齐平（A4−F1=+1.6/−0.4/−0.6pp）。→ delay T80/task0 的固定 kernel 收益主要是**收敛速度/early advantage**，不是最终可达质量。
- **初始化是有效杠杆**：init-burst5/decay-slow 在 65ep 比 A1 高 **+6.5/+6.1pp**，且终点仍在初始核族（burst5-占优 0.62–0.81；decay-slow-占优 0.58–0.79），并未回到 single。预注册“停留”阈值（Δα_L1<0.2）过严，实际是“软化但不离族”。
- **shared 1-D λ 不动**：|Δλ|≤0.034（500ep 仍如此）；初始条件 landscape 平（gap≈0），full-BPTT 初始 ∂L/∂λ 量级 1e-8–1e-6、batch 内符号一致。由于 500ep 时 single 已收敛到 F1 水平，本轮**不把“不移动”判为失败**。
- **协议核对**：截断通道对 scheduler 梯度恒为 0（detach 在每个 5 步边界，损失只连最后 5 步），scheduler 必须走 hybrid full。
- 预注册 Q1/Q2/Q3 判定与决策映射见 [REPORT.md](REPORT.md) §5；本轮证据更支持先做 **budget-matched re-baseline**（T≥80 的 fixed kernel 在 500ep 是否仍领先），再决定 continuous λ / exploration 主实验。

## 运行

```bash
# 协议核验（几秒）
FLOW_THREADS=1 python verify_audit.py

# 主矩阵（3 seeds；A1 与 V24 bank_delay 逐位一致；本机 --jobs 3 约 73 min）
FLOW_THREADS=1 python audit.py --T 80 --tasks 0 --seeds 11 22 33 --jobs 3 --out results/audit
python analyze.py   # 生成 REPORT.md / results/audit/summary.json

# smoke（只验证代码路径，结果不用于判定）
FLOW_THREADS=1 python audit.py --arms A1 A2 B1 F1 --epochs 2 --seeds 11 --out smoke
```

注意：65ep fixed 锚点直接读 V24 `results/delay/`；条件 landscape 冻结当前 core，只作局部几何证据，联合可达性看 V24 Phase A 与 F* arms。Phase 0 结论只针对 T=80/task0，不外推到 T=160 与 Chain-select。

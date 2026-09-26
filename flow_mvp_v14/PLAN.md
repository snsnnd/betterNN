# 第十四轮：Write 因果归因

## 背景

第十三轮发现：在 5 步截断下 Write 门梯度恒为 0（输入只在 t=0/1 写入，梯度窗口只覆盖最后 5 步），第七轮以来 Write 实际是随机固定的 task-conditioned 输入调制。本轮把它作为唯一变量，判定 Write 是否有必要存在。

## 设计

固定 A'B'C'D'、Flow（N=256）、o0 顺序、65 轮/阶段、每步 128 样本、5 种子、r∈{0,12.5%}、缓冲 32 样本/任务。唯一变化是 Write：

| 模式 | 定义 |
|---|---|
| `frozen` | 现状：随机初始化的 Write MLP，5 步截断下不被训练 |
| `const1` | p ≡ 1（完全写入） |
| `const05` | p ≡ 0.5 |
| `full_bptt` | 去掉 detach，完整 BPTT，Write 真实训练（整网梯度协议随之改变） |
| `aux` | 保持 5 步截断，另在 t=5 边界对状态加辅助 BCE（λ=0.5），使 Write 获得局部梯度 |

共 5 模式 × 2 比例 × 5 种子 = 50 条流。

## 判定

- `const* ≈ frozen` → 随机 task-conditioned 调制没有结构贡献，Write 可删除。
- `const* < frozen` → 随机调制有贡献，需保留某种输入调制。
- `full_bptt / aux > frozen` → 训练 Write 有收益，值得修复协议后保留。
- 以上均与 r=0 和 r=12.5% 两个条件一起看。
- 回归核对：`frozen` 路径应与第十一轮 o0/all 完全一致。

## 边界

- `full_bptt` 改变整网梯度协议，不只是 Write；`aux` 的 λ 与边界位置是可调选择。
- 只测 o0 顺序与单一规模，结论限于本 benchmark。

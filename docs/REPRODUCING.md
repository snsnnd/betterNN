# 运行与复现指南

[返回项目首页](../README.md)

## 1. 环境

早期实验建议使用 Python 3.10+ 和独立虚拟环境；第五～七轮固定的 NumPy 2.3.5 要求 Python 3.11+，原第一轮运行记录使用 Python 3.12.14。现有训练脚本无需下载外部数据。各轮依赖以自己的 `requirements.txt` 为准：

| 目录 | 依赖声明 |
|---|---|
| 第1～3轮 | `torch>=2.2`、`numpy>=1.26`、`matplotlib>=3.8` |
| 第4轮 | 上述依赖，加 `scipy>=1.11`、`networkx>=3.0` |
| 第5～7轮 | `torch==2.14.0+cpu`、`numpy==2.3.5`、`matplotlib` |
| 第8轮 | `torch==2.14.0`、`numpy==2.4.6`、`matplotlib==3.11.2`（uv 环境，CUDA 13） |
| 第9轮 | `torch==2.14.0`、`numpy==2.4.6`（同 .venv） |
| 第10轮 | 同上；轻量实验用 CPU 更快 |

第五～七轮的精确版本是归档声明，是否可从当前安装源取得需以实际安装结果为准。改用其他版本可能无法通过代码中的逐值相等断言。

第八、九轮使用项目根目录的 `.venv`（由 uv 创建，WSL/Linux，Python 3.11，复用 uv 缓存中的 GPU torch），在 RTX 4060 上运行；第十～十四轮等轻量实验按经验改在 CPU 运行（N=256 时 CPU 明显快于 GPU，且可多进程并行）：

```bash
uv venv /mnt/d/betterNN/.venv --python 3.11
uv pip install --python /mnt/d/betterNN/.venv/bin/python -r flow_mvp_v8/requirements.txt
```

该 `.venv` 是 Linux 虚拟环境，不能与 Windows 侧的 `vision_work/.venv` 混用；Windows venv 无法被 WSL 的 Linux Python 直接复用。

以下实验命令均要求终端工作目录为对应的 `flow_mvp*` 目录。先运行：

```bash
python -m pip install -r requirements.txt
```

不要把根目录当成所有脚本的统一运行目录；部分代码使用 `Path('results')` 和同目录模块导入。

## 2. 最小体验：加载已有模型

工作目录：`flow_mvp/`。

```bash
python predict.py
```

这会加载已有检查点演示同一段信号在不同任务指令下的预测。第2、3轮也提供各自的 `predict.py`。

若要实际训练一个最小核心实验，在 `flow_mvp/` 中执行：

```bash
python experiment.py --variants modulated --epochs 100 --seeds 11 --out quick_run
python predict.py --checkpoint quick_run/modulated_seed11.pt
```

`quick_run/` 会在当前目录生成。它是单配置单种子结果，不具备原完整三种子实验的统计意义。

## 3. 各轮主要入口

### 第1轮

工作目录：`flow_mvp/`。

```bash
python experiment.py --epochs 100 --seeds 11 22 33 --out results_reproduced
```

验证原归档的 24 个模型：`python verify_results.py`。该脚本固定读取 `results/`，没有 `--out` 参数。`python make_plot.py` 同样读取原默认结果目录。

### 第2轮

工作目录：`flow_mvp_v2/`。

```bash
python experiment.py --epochs 100 --seeds 11 22 33 --out results_reproduced
python experiment.py --variants marker_slowleak --epochs 100 --seeds 11 22 33 --out results_slowleak_reproduced
python verify.py --out results_reproduced
python verify.py --out results_slowleak_reproduced
```

`python verify.py --data-only` 只做数据语义、梯度和冻结检查。`python make_plot.py` 读取默认 `results/` 与 `results_slowleak/`，不会自动使用上面新建的输出目录。

### 第3轮

工作目录：`flow_mvp_v3/`。

```bash
python experiment.py --epochs 100 --seeds 11 22 33 44 55 --out results_reproduced
python verify.py --out results_reproduced
```

只训练简化模型可用：

```bash
python experiment.py --variants g101 --seeds 11 --out quick_run
python predict.py --checkpoint quick_run/g101_seed11.pt
```

完整实验是 15 配置 × 5 种子，共 75 个检查点。分析入口为 `analyze.py`，报告生成入口为 `build_report.py`。

### 第4轮：按分支操作

工作目录：`flow_mvp_v4/`。先阅读[分支说明](EXPERIMENTS.md)，分别在工作副本中运行，避免两个分支争用 `results/salience/`。

局部目的路由和辅助监督分支的训练入口：

```bash
python routing.py --out results/routing_reproduced
python salience.py --out results/salience_reproduced
```

这两个入口支持 `--epochs`、`--seeds`、`--out`。`make_report.py` 仍固定读取默认路径，不能直接接续以上自定义目录。当前归档默认 Salience 指标使用探索分支字段，直接运行 `make_report.py` 会遇到字段不匹配，不能将其视为开箱可运行的统一报告链。

受控拓扑与延迟规则探索分支，在独立工作副本中的原命令为：

```bash
python route_analysis.py
python experiments.py --study topology
python experiments.py --study salience
python experiments.py --study delay
python experiments.py --study delay_block
python verify.py
python analyze.py
python extra_analysis.py
```

当前 `verify.py` 导入 `experiments.py`，核验这条探索分支；它不是当前 `verification.txt` 中 36 模型分支日志的同一版核验程序。

### 第5轮

工作目录：`flow_mvp_v5/`。

```bash
python run.py
python finish.py
python diagnostics.py
python plot.py
```

共 36 次训练。`run.py` 固定写入本目录的 `results/`，没有输出目录参数；`finish.py` 重载模型、复算指标并重写 `REPORT.md` 和 `verification.txt`。`run.py` 在被导入时也会写入 `results/config.json`。

### 第6轮

工作目录：`flow_mvp_v6/`。

```bash
python run.py
python finish.py
python plot.py
```

依赖已经提供的 `initial/` 预训练检查点。持续学习为 12 条流、36 个适应阶段，另有 9 个独立适应对照。正常 W 的 Router 补充实验入口：

```bash
python attribute.py
python finish_attribute.py
```

这两条实验线分别保存到 `results/` 与 `attribution/`，前者是持续学习，后者是从头训练的 Router 对照。

## 4. 第七轮：容量与分区

### 只查看已保存结果

直接打开以下文件即可，无需模型运行环境：

- [REPORT.md](../flow_mvp_v7/REPORT.md)：完整结论。
- [summary.json](../flow_mvp_v7/summary.json)：10 个规模/模式组合的均值和标准差。
- [all_metrics.json](../flow_mvp_v7/all_metrics.json)：每种子的完整分析记录。
- [scaling.png](../flow_mvp_v7/scaling.png)：结果图。
- [results/](../flow_mvp_v7/results/)：每条流的原始结果与最终模型。

### 训练与分析顺序

工作目录：`flow_mvp_v7/`。

```bash
python experiment.py
python joint_control.py
python analyze.py
python plot.py
```

`experiment.py` 跑 50 条顺序学习流；`joint_control.py` 跑 5 个混合任务参考。完整归档已有 `joint_control/` 结果，分析可以直接读取；若从空结果开始，两部分都必须完成，`analyze.py` 才具备全部输入。

`joint_control.py` 不会检查已有结果并跳过，重跑会覆盖自己的 5 组产物。

### 筛选运行

```bash
python experiment.py --sizes 64 128 --seeds 11 --modes dense topk
```

支持的原设计规模为 64、128、256、512、1024，模式为 `dense` / `topk`。`config.json` 写的是全部预设，不随筛选参数缩小，因此实际完成范围需检查结果文件及其中的 `n/mode/seed`。

当前已有结果时，上面命令通常会跳过对应记录，而不是重新训练。独立重现实验应使用项目工作副本，并在副本中将原 `results/` 改名保存，再执行训练。脚本没有 `--out` 或 `--force` 参数。

`analyze.py` 要求完整 50 条主结果，并读取全部 5 个混合训练结果；部分训练完成后不能直接用它生成完整报告。`finalize_when_ready.py` 只轮询主结果 JSON 数量达到 50，并不检查混合训练结果是否就绪。

### 续跑规则

`run(n, mode, seed)` 发现 `results/{mode}_{n}_{seed}.json` 已存在就跳过。判断只看 JSON，不会检查 `.pt` 是否仍存在，也不支持从阶段中间或优化器状态恢复。

结果 JSON 尚未写出的中断流会从头重跑。历史续跑命令见 [RESUME.md](../flow_mvp_v7/RESUME.md)：

```bash
FLOW_THREADS=4 python experiment.py --sizes 1024 --seeds 22 33 44 55
```

该语法适用于 Bash。默认 `FLOW_THREADS=1`；历史前 42 条流为 1 线程，补跑 8 条为 4 线程。新记录带 `threads` 字段，旧记录缺失时分析按 1 线程处理。

### 输出与副作用

| 文件 | 内容 |
|---|---|
| `config.json` | 全部预设规模、种子、训练预算及运行时torch版本 |
| `results/{mode}_{n}_{seed}.json` | 4×4成绩矩阵、曲线、初始/最终诊断、训练耗时和参数量 |
| `results/{mode}_{n}_{seed}.pt` | 最终 `state`、`n`、`mode`、`seed` |
| `joint_control/{seed}.json` / `.pt` | 混合训练分数及模型 |
| `summary.json` / `all_metrics.json` | 正式聚合结果及补充诊断 |
| `joint_summary.json` | 混合训练参考汇总 |
| `REPORT.md` / `verification.txt` | `analyze.py` 生成的报告和核验记录 |
| `scaling.png` | `plot.py` 输出 |

`analyze.py` 会重写报告；其生成模板不包含当前归档 `REPORT.md` 最后的“本轮结论”补充段落。第5、6轮 `finish.py` 也有类似的重写行为。需要保留原解释文本时，使用工作副本执行这些脚本。

## 5. 第八轮：Top-K 机制归因

工作目录：`flow_mvp_v8/`。环境为项目根目录 `.venv`（见第 1 节）。与第七轮不同，第八轮脚本支持 `--out`、`--device` 等筛选参数。

```bash
cd flow_mvp_v8
/mnt/d/betterNN/.venv/bin/python experiment.py            # 15 条流，默认 cuda，自动跳过已有结果
/mnt/d/betterNN/.venv/bin/python parity_check.py          # 可选：CPU 下单条流与第七轮存档逐位比对
/mnt/d/betterNN/.venv/bin/python analyze.py               # 重载 15 个检查点、复算、写 REPORT/verification
/mnt/d/betterNN/.venv/bin/python plot.py                  # attribution.png
```

实际运行记录：15 条流在 RTX 4060 上约 6 分钟；`analyze.py` 的核验通过；GPU 结果与第七轮 CPU 存档的离散成绩矩阵 5/5 一致，CPU 单条复跑与存档逐位一致（内部浮点存在设备差异）。

筛选与冒烟示例：

```bash
/mnt/d/betterNN/.venv/bin/python experiment.py --seeds 11 --modes learned_topk --epochs 2 --out smoke --device cpu
```

注意：

- `experiment.py` 同样按结果 JSON 是否存在跳过；独立复现需在工作副本中进行。
- `analyze.py` 要求恰好 15 条结果（3 模式 × 5 种子），部分运行无法生成完整汇总。
- 输出：`results/{mode}_256_{seed}.json` / `.pt`、`summary.json`、`all_metrics.json`、`paired_differences.json`、`REPORT.md`、`verification.txt`、`attribution.png`。
- `analyze.py` 会重写 `REPORT.md` 与 `verification.txt`，内容完全由脚本生成。

## 6. 第九轮：遗忘定位

工作目录：`flow_mvp_v9/`。协议为先全参数学习 A，再对 B/C/D 分别保护 W/控制器/读出并加入固定预算 replay。

```bash
cd flow_mvp_v9
/mnt/d/betterNN/.venv/bin/python experiment.py      # 5 预训练 + 105 阶段 + 15 对照
/mnt/d/betterNN/.venv/bin/python analyze.py         # 重载 125 个检查点、复算、写 REPORT/verification
/mnt/d/betterNN/.venv/bin/python plot.py            # forgetting.png
```

实际运行记录：训练约 10 分钟，分析核验约 2 分钟；所有阶段的冻结参数组在检查点中与预训练起点逐位相同。

筛选与冒烟示例：

```bash
/mnt/d/betterNN/.venv/bin/python experiment.py --arms all replay freeze_W --seeds 11 --epochs 2 --out smoke --device cpu --skip-controls
/mnt/d/betterNN/.venv/bin/python analyze.py --out smoke --seeds 11 --arms all replay freeze_W
```

注意：

- `--out smoke` 时报告与汇总写入 `smoke/`，不会覆盖主结果；默认 `results/` 时写入轮次根目录。
- 阶段检查点命名 `results/{arm}_{seed}_stage{task}.pt`；预训练在 `results/pretrain/`；对照为 `results/control_{seed}_{task}.pt`。
- `analyze.py` 需要完整的“种子数 × 组数”结果，默认要求 5 种子 × 7 组，并核对 15 个对照。
- 同样按 JSON 是否存在跳过训练；独立复现需使用工作副本。

## 7. 第十轮：benchmark 校准

工作目录：`flow_mvp_v10/`。新 benchmark A'B'C'D'，先用验证集选训练预算，再跑顺序 CL 基线。全部在 CPU 运行。

```bash
cd flow_mvp_v10
/mnt/d/betterNN/.venv/bin/python experiment.py --modes pretrain scratch adapt --epochs 150 --out results --device cpu
/mnt/d/betterNN/.venv/bin/python experiment.py --modes cl --epochs 65 --out results --device cpu
/mnt/d/betterNN/.venv/bin/python analyze.py --epochs 150    # 复算、写 benchmark.json 与 REPORT.md
/mnt/d/betterNN/.venv/bin/python plot.py                    # benchmark.png
```

实测：校准约 9 分钟；E90=48、E95=65；CL 基线 acquisition 97.37%、最终平均 80.80%、遗忘 22.10 个百分点。`results_legacy_calibration/` 保留了旧互补任务集的校准结果（D 无法稳定 adapt 的失败证据）。

注意：

- `analyze.py --epochs` 指校准曲线的最大长度（150），CL 阶段预算记录在结果行内（65）。
- 第十一轮在此 benchmark 上运行 replay × 参数保护矩阵；旧任务集继续保留在第八、九轮。

## 8. 第十一轮：replay × 参数保护

工作目录：`flow_mvp_v11/`。150 条流（2 组 × 5 比例 × 3 顺序 × 5 种子），可三个顺序并行，每进程单线程（32 核机器上约 30 分钟跑完）。

```bash
cd flow_mvp_v11
for o in o0 o1 o2; do
  /mnt/d/betterNN/.venv/bin/python experiment.py --orders $o --out results --device cpu > logs/$o.log 2>&1 &
done
wait
/mnt/d/betterNN/.venv/bin/python analyze.py    # 600 个阶段检查点复算 + REPORT.md
/mnt/d/betterNN/.venv/bin/python plot.py       # replay_matrix.png
```

实测：150 条流并行约 30 分钟；核验约 4 分钟；o0+r0+all 与第十轮基线 5/5 一致。

注意：

- 回放比例用 `--ratios`，例如 `--ratios 0 0.25`；顺序用 `--orders o0 o1 o2`。
- 冒烟：`--epochs 2 --seeds 11 --ratios 0 .25 --orders o0`。
- 结果按流保存：`results/{order}_{arm}_r{label}_{seed}.json` 与 `_stage{0..3}.pt`。

## 9. 第十二轮：架构等预算 replay

工作目录：`flow_mvp_v12/`。180 条流（3 架构 × 4 比例 × 3 顺序 × 5 种子），三顺序可并行。

```bash
cd flow_mvp_v12
for o in o0 o1 o2; do
  /mnt/d/betterNN/.venv/bin/python experiment.py --orders $o --out results --device cpu > logs/$o.log 2>&1 &
done
wait
/mnt/d/betterNN/.venv/bin/python analyze.py
/mnt/d/betterNN/.venv/bin/python plot.py
```

实测：180 条流并行完成；核验 720 个阶段检查点约 4 分钟；flow 路径与第十一轮 all 组 20/20 逐位一致。

## 10. 第十三轮：replay 保护机制

工作目录：`flow_mvp_v13/`。梯度冲突与组件移植复用第十一轮检查点，无需重训；缓冲扫描新增 30 条流。

```bash
cd flow_mvp_v13
/mnt/d/betterNN/.venv/bin/python gradient_conflict.py
/mnt/d/betterNN/.venv/bin/python transplant.py
for M in 1 2 4 8 16 32; do
  /mnt/d/betterNN/.venv/bin/python buffer_scan.py --orders o0 --archs flow --ratios .125 \
    --buffer-per-task $M --out results_buffer --device cpu > logs/m$M.log 2>&1 &
done
wait
/mnt/d/betterNN/.venv/bin/python analyze.py
/mnt/d/betterNN/.venv/bin/python plot.py
```

实测：冲突 27 秒、移植 51 秒、缓冲扫描并行约 3 分钟；M=32 与第十一轮 r=12.5% 逐位一致。

## 11. 第十四轮：Write 因果归因（最新）

工作目录：`flow_mvp_v14/`。5 种写入模式 × 2 比例 × 5 种子，可 5 进程并行。

```bash
cd flow_mvp_v14
for w in frozen const1 const05 full_bptt aux; do
  /mnt/d/betterNN/.venv/bin/python experiment.py --write-modes $w --ratios 0 .125 \
    --out results --device cpu > logs/$w.log 2>&1 &
done
wait
/mnt/d/betterNN/.venv/bin/python analyze.py
/mnt/d/betterNN/.venv/bin/python plot.py
```

实测：50 条流并行约 20 分钟；核验 200 个阶段检查点；frozen 与第十一轮逐位一致 10/10。

## 12. 原归档校验与打包

在项目根目录执行：

```bash
sha256sum --quiet -c SHA256SUMS.txt
```

该命令核对清单中记录的原归档文件；成功时没有输出。它验证文件完整性，不验证模型效果。原清单没有包含本次新增的 `README.md` 和 `docs/`。

根目录打包入口：

```bash
python build_all_experiments.py
```

实际行为是：收集七轮目录、可用的历史 ZIP 和打包脚本，生成 `all_experiments_v1-v7.zip`，包内目录为 `all_experiments/`，并在 ZIP 内生成 README、manifest、counts、校验文件。脚本还执行 ZIP CRC 与包内 SHA-256 检查。

当前脚本不收集新增的根 `README.md` 或 `docs/`，也不会更新根目录已有索引文件。若要交付本次整理后的文档，需要同时包含这些新增文件。

## 13. 常见问题

| 现象 | 核查方向 |
|---|---|
| 找不到 `results/` 或本地模块 | 是否在对应轮次目录运行 |
| 找不到指定的 torch 安装版本 | 核对该轮固定版本与所使用的软件源；版本差异需记录 |
| 第七轮训练立即结束 | 对应结果 JSON 已存在，触发跳过 |
| 第七轮分析提示不是50条 | 主实验未完整完成，或混入了其他主结果文件 |
| 第七轮分析找不到混合训练文件 | 从空结果开始时尚未运行 `joint_control.py` |
| 复算逐值相等断言失败 | 检查 PyTorch、设备、线程、数据种子和检查点是否匹配 |
| 第九轮 replay 报“单任务 batch”断言 | 该断言只存在于第八轮；第九轮已移除以支持混合 batch |
| 第四轮报告生成缺少 `kind` 等字段 | 两套分支的 Salience 指标格式不同，核对分支与输出目录 |
| 分析后原报告末尾消失 | 生成脚本重写报告，未包含归档中的后补结论 |

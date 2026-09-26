# 容量与分区实验
阅读 REPORT.md（结果）、PLAN.md（设计和边界）、summary.json（均值和样本标准差）。

运行 python experiment.py，完成后 python analyze.py 和 python plot.py。
支持 --sizes 64 128 --seeds 11 --modes dense topk 筛选运行；完整配置仍记录全部预设值。已存在单条结果JSON则跳过，删除该条对应结果后可重跑；不要把部分运行当作完整50组。

results保留最终检查点、各阶段成绩、逐轮训练/验证曲线、初始化与最终节点/路由统计。每规模每模式5种子，共50条顺序学习流。只保存最终模型，中间阶段有指标没有检查点。训练没有自动挑选测试最佳结果。

Top-K前向严格每角色16节点，共64个；反向是straight-through代理。仍为稠密运算，不承诺节省FLOPs。任务ID从t0可见，规则R从t4；不同于前轮。低活动集重叠不能单独证明学习出了脑区，注意训练前已经存在的随机分区。

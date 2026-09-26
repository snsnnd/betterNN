第五轮：反向传播窗口机制筛查

先阅读 REPORT.md 与 PLAN.md。运行 python run.py，再运行 python finish.py、python diagnostics.py、python plot.py。

base_routing.py 是第四轮局部模块实现的原样快照；run.py 只为状态增加梯度截断，并控制底层学习率。运行不会依赖此前目录。window=0 表示完整反向传播。所有36个检查点与每轮训练损失/验证分数在 results 中。

统计只有3个随机种子，不能解释为通用架构排名；非等参数、非等计算预算比较。第4步后规则持续可见，任务只输出两位，截断时无早期辅助损失。没有测持续学习遗忘。

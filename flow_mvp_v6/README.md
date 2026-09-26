# 第六轮交付
REPORT.md：预训练后A→B→C→D持续学习及遗忘。
ATTRIBUTION_REPORT.md：正常更新W时动态Router、静态Router、通路全开比较。
PLAN.md：设计、收到补充意见后新增的对照及边界。

运行顺序：python run.py；python finish.py；python plot.py。
Router补充：python attribute.py；python finish_attribute.py。
需要PyTorch与NumPy，绘图额外matplotlib。

initial保存第五轮full预训练起点，results保存持续学习检查点和逐轮曲线；attribution保存从头训练Router对照，dynamic_v5_metrics.json为已验证的第五轮动态组参考数据。两项研究不要混作同一排行榜。窗口模型实现复用第五轮快照。没有持续学习回放或局部可塑性；预训练成本不能忽略。没有等参数预算匹配。

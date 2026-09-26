from pathlib import Path
import hashlib,json,zipfile,collections
ROOT=Path(__file__).parent
folders=['flow_mvp']+[f'flow_mvp_v{i}' for i in range(2,8)]
entries=[]
for folder in folders:
 for p in sorted((ROOT/folder).rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc','.pyo']:
   entries.append((p,p.relative_to(ROOT).as_posix()))
for folder in folders[:-1]:
 p=ROOT/(folder+'.zip')
 if not p.exists():p=ROOT/'historical_deliveries'/(folder+'.zip')
 if p.exists():entries.append((p,'historical_deliveries/'+p.name))
entries.append((ROOT/'build_all_experiments.py','build_all_experiments.py'))
manifest=[];counts={}
for p,name in entries:
 data=p.read_bytes();manifest.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
for folder in folders:
 members=[(p,n) for p,n in entries if n.startswith(folder+'/')]
 counts[folder]={'files':len(members),'python_files':sum(p.suffix=='.py' for p,_ in members),'checkpoints':sum(p.suffix=='.pt' for p,_ in members),'bytes':sum(p.stat().st_size for p,_ in members)}
readme='''# 大网＋调制器实验总包：第1～7轮

先读 flow_mvp_v7/REPORT.md 查看最新结果。完整原始指标、逐轮曲线、已保存的检查点、代码和各轮报告均保留。未删掉失败种子或负结果。

| 目录 | 主要内容 |
|---|---|
| flow_mvp | 最小固定动力网、调制器与传统网络对照 |
| flow_mvp_v2 | 有效标记、保持门、位置与长度泛化 |
| flow_mvp_v3 | 三门消融、标记可靠性、底层更新方式 |
| flow_mvp_v4 | Salience、条件路由、拓扑与探索分支 |
| flow_mvp_v5 | Full/10/5步BPTT × 冻结/慢速/正常W，与GRU对照 |
| flow_mvp_v6 | 顺序持续学习与遗忘、动态/静态/全开Router归因 |
| flow_mvp_v7 | 64～1024节点 × dense/Top-K64 × 5种子，及混合训练参考 |
| historical_deliveries | 第1～6轮当时交付ZIP原件，保留早期版本，避免覆盖历史 |

## 代码与数据是什么
- *.py：训练、合成数据生成、分析、验证、绘图脚本。
- metrics/config/history/summary等JSON、CSV、NPZ：已保存的实验分数、配置、曲线和轨迹；各轮文件名略有不同。
- *.pt：保存下来的模型权重；有些轮次只保存最终或验证最优模型，并未保存每轮/每阶段模型。
- REPORT/PLAN/README/verification及日志：报告、设计、运行和验证记录。
- MANIFEST.json：本包全部项目文件的路径、大小和SHA-256；COUNTS.json为按轮统计。

训练/验证/测试输入是合成数据，由各轮数据生成器及随机种子生成。大部分输入张量当时未单独落盘；本包包含生成代码与种子，不能把未落盘张量说成已打包的原始采集数据。已保存的数值结果和轨迹全部纳入。没有现实采集数据集。

## 复现
每轮先读该目录的README、PLAN与requirements；在对应目录运行训练和验证入口。环境版本以各轮config及requirements为准。不同PyTorch/线程/设备可能产生数值差异。
第七轮完整命令：python experiment.py；python analyze.py；python plot.py。
第七轮续跑细节见RESUME.md：42条已有完整流保留，1024节点剩余8条续跑；恢复后每条记录线程数，旧记录按1线程。
第七轮训练50条顺序学习流，每条4阶段，另5个混合训练模型。只保存50个最终模型，未声称有200个中间阶段检查点。

## 历史分支说明
第四轮目录保留了两套研究路径：routing.py/salience.py对应局部目的端口和辅助监督报告；experiments.py及topology/delay/delay_block等结果属于另外的探索路径。它们的数据、架构和协议不同，不能把分数合并成一个排行榜。旧README与后续REPORT可能面向不同路径，已原样保留以便追溯。

## 结论边界
门控架构、任务可见信息和训练预算在各轮有变化，不能直接把跨轮准确率当成提升曲线。第七轮Top-K是有任务身份、人工角色模块和代理梯度的选择机制；64节点是逻辑活跃上限，仍使用稠密计算，不是固定FLOPs的稀疏加速证明。

SHA256SUMS.txt覆盖所有项目文件及索引，解压后可校验文件是否完整。压缩包已做CRC检查。
'''
metadata={'README_START_HERE.md':readme.encode(),'MANIFEST.json':json.dumps(manifest,ensure_ascii=False,indent=2).encode(),'COUNTS.json':json.dumps(counts,indent=2).encode()}
checks=[f"{x['sha256']}  {x['path']}" for x in manifest]
checks += [f'{hashlib.sha256(v).hexdigest()}  {k}' for k,v in metadata.items()]
metadata['SHA256SUMS.txt']=('\n'.join(checks)+'\n').encode()
archive=ROOT/'all_experiments_v1-v7.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p,name in entries:z.write(p,'all_experiments/'+name)
 for name,data in metadata.items():z.writestr('all_experiments/'+name,data)
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 for item in manifest:assert hashlib.sha256(z.read('all_experiments/'+item['path'])).hexdigest()==item['sha256']
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'files':len(entries)+len(metadata),'counts':counts},indent=2))

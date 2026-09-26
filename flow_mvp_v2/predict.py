"""加载第二轮模型，显示标记变化和延迟下的预测。"""
import argparse,torch
from experiment import build,dataset

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',default='results/marker_memory_seed11.pt');args=ap.parse_args();torch.set_num_threads(2)
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=True);model=build(ckpt['variant']);model.load_state_dict(ckpt['state_dict']);model.eval()
    x,q,m,y=dataset(4,20260925)
    with torch.no_grad():p=model(x,q,m).sigmoid()
    for i in range(4):
        print(f'样本{i}：目标通道{q[i].argmax().item()}，有效时刻{m[i].nonzero().flatten().tolist()}，真实标签{int(y[i])}，预测概率{p[i].item():.3f}，预测标签{int(p[i]>.5)}')
    print('这只是固定示例；总体表现和分布外测试见 REPORT.md。')
if __name__=='__main__':main()

"""加载真实训练权重，演示同一组信号在不同任务指令下的输出。"""
import argparse
import torch
from experiment import FlowNet, Baseline

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default='results/modulated_seed11.pt'); args=ap.parse_args()
    torch.set_num_threads(2)
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=Baseline(ckpt['variant']) if ckpt['variant'] in ['rnn','mlp'] else FlowNet(ckpt['variant'])
    model.load_state_dict(ckpt['state_dict']);model.eval()
    gen=torch.Generator().manual_seed(20260925)
    x=torch.randn(1,8,4,generator=gen).expand(4,-1,-1)
    q=torch.eye(4)
    with torch.no_grad(): probs=model(x,q).sigmoid()
    print('同一段四路信号，分别询问每一路前四步总和是否为正：')
    for i,p in enumerate(probs):
        s=x[i,:4,i].sum().item()
        print(f'通道 {i}: 真实总和={s:+.3f}, 真实标签={int(s>0)}, 预测概率={p.item():.3f}, 预测标签={int(p>.5)}')
if __name__=='__main__':main()

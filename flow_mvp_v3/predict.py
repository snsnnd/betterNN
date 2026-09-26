import argparse,torch
from experiment import build,dataset,inputs,cue_mode

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',default='results/g101_seed11.pt');args=ap.parse_args();torch.set_num_threads(1)
 ck=torch.load(args.checkpoint,map_location='cpu',weights_only=True);model=build(ck['variant']);model.load_state_dict(ck['state_dict']);model.eval()
 raw=dataset(4,20260925);x,q,m,y=inputs(raw,cue_mode(ck['variant']),1123)
 with torch.no_grad():probs=model(x,q,m).sigmoid()
 for i,p in enumerate(probs):print(f'样本{i}，通道{q[i].argmax().item()}，真实有效时刻{raw[2][i].nonzero().flatten().tolist()}，标签{int(y[i])}，预测{int(p>.5)}，概率{p.item():.3f}')
if __name__=='__main__':main()

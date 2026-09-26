"""重新加载全部检查点，复算测试准确率，并确认冻结权重与初始化一致。"""
import json
from pathlib import Path
import torch
from experiment import FlowNet, Baseline, dataset, accuracy

torch.set_num_threads(2)
rows=json.loads(Path('results/metrics.json').read_text())
assert len(rows)==24
for r in rows:
    seed,v=r['seed'],r['variant'];torch.manual_seed(seed)
    model=Baseline(v) if v in ['mlp','rnn'] else FlowNet(v)
    initial={k:t.clone() for k,t in model.state_dict().items()}
    ckpt=torch.load(f'results/{v}_seed{seed}.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(ckpt['state_dict']);model.eval()
    observed=accuracy(model,dataset(4096,seed*100+2))
    assert observed==r['test_accuracy'],(v,seed,observed,r['test_accuracy'])
    if isinstance(model,FlowNet):
        for k in ['B','C','mask']+([] if v=='slow_backbone' else ['W']):
            assert torch.equal(model.state_dict()[k],initial[k]),(v,seed,k)
        if v=='controller_only':
            for k in ['head.weight','head.bias']:assert torch.equal(model.state_dict()[k],initial[k])
print('PASS: 24 checkpoints reproduce test metrics exactly; all required frozen tensors unchanged.')

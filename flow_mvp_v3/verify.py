"""核验三门实现、任务可辨识性、冻结权重和保存模型。"""
import argparse,json,itertools
from pathlib import Path
import torch
from experiment import build,Flow,VARIANTS,cue_mode,corrupt,dataset,get_tests,accuracy,assert_frozen,markerless_reference
from v2_reference import FlowNet as V2

def structural_checks():
    x,q,m,y=dataset(64,777)
    assert torch.all(m.sum(1)==2)
    truth=(x*m[:,:,None]*q[:,None,:]).sum((1,2))>0
    assert torch.equal(truth,y.bool())
    assert torch.equal(corrupt(m,'none',1),torch.zeros_like(m))
    assert torch.equal(corrupt(m,'noisy',2),corrupt(m,'noisy',2))
    # 无标记时：同一x和q，不同合法标记确实可能有相反答案。
    values=x[0,:6,q[0].argmax()];answers=[bool(values[i]+values[j]>0) for i,j in itertools.combinations(range(6),2)]
    assert any(answers) and not all(answers)
    # 与第二轮完整模型逐元素一致，保证不是更换动力学取得收益。
    torch.manual_seed(11);a=build('g111')
    torch.manual_seed(11);b=V2('marker_memory')
    assert all(torch.equal(a.state_dict()[k],v) for k,v in b.state_dict().items())
    assert torch.equal(a(x,q,m),b(x,q,m))
    for v in VARIANTS:
        torch.manual_seed(11);model=build(v)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x,q,m),y);loss.backward()
        if isinstance(model,Flow):
            for module,enabled in [(model.upper,model.I),(model.lower,model.R),(model.update,model.H)]:
                assert all(p.requires_grad==enabled for p in module.parameters())
                if enabled:assert sum(p.grad.abs().sum().item() for p in module.parameters())>0
                else:assert all(p.grad is None for p in module.parameters())
            assert (model.W.grad is not None)==model.W.requires_grad
    print('PASS: label rules, marker removal, non-identifiability example, V2 equivalence, all gate gradients and freeze flags.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-only',action='store_true');ap.add_argument('--out',default='results');args=ap.parse_args();torch.set_num_threads(1)
    structural_checks()
    if args.data_only:return
    root=Path(args.out);rows=json.loads((root/'metrics.json').read_text());config=json.loads((root/'config.json').read_text())
    assert len(rows)==len(config['seeds'])*len(config['variants'])
    count=0
    for r in rows:
        seed,v=r['seed'],r['variant'];torch.manual_seed(seed);model=build(v);initial={k:t.clone() for k,t in model.state_dict().items()}
        ck=torch.load(root/f'{v}_seed{seed}.pt',weights_only=True,map_location='cpu');model.load_state_dict(ck['state_dict']);model.eval();assert_frozen(model,initial)
        _,tests=get_tests(seed,cue_mode(v))
        for name,data in tests.items():
            observed=accuracy(model,data);assert observed==r[name],(seed,v,name,observed,r[name]);count+=1
    print(f'PASS: {len(rows)} checkpoints and {count} saved accuracy entries exactly reproduced; frozen tensors unchanged.')
if __name__=='__main__':main()

"""语义检查与已保存模型的独立复算。--data-only 可在训练期间执行。"""
import argparse,json
from pathlib import Path
import torch
from experiment import CONDITIONS,FlowNet,build,dataset,evaluate,assert_frozen

def check_data():
    for i,(name,cfg) in enumerate(CONDITIONS.items()):
        x,q,m,y=dataset(1024,710+i,**cfg)
        assert torch.all(m.sum(1)==2)
        # 用不同表达式实现已知任务，核验标签。
        selected=x.gather(2,q.argmax(1)[:,None,None].expand(-1,x.shape[1],1)).squeeze(-1)
        truth=torch.stack([selected[j,m[j].bool()].sum() for j in range(len(x))])>0
        assert torch.equal(truth,y.bool())
        if cfg['region']=='early':assert m[:,6:].sum()==0
        if cfg['region']=='late':assert m[:,:-6].sum()==0
        assert .4<y.mean()<.6
        a=dataset(32,778,**cfg);b=dataset(32,778,**cfg)
        assert all(torch.equal(u,v) for u,v in zip(a,b))
    x,q,m,y=dataset(32,111)
    model=FlowNet('marker_memory')
    logits=model(x,q,m);torch.nn.functional.binary_cross_entropy_with_logits(logits,y).backward()
    assert model.W.grad is None
    for name,mod in [('upper',model.upper),('lower',model.lower),('head',model.head),('update',model.update)]:
        assert sum(p.grad.abs().sum().item() for p in mod.parameters())>0,name
    print('PASS: task labels, marker count/regions, deterministic generation, controller gradients, frozen W.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-only',action='store_true');ap.add_argument('--out',default='results');args=ap.parse_args();torch.set_num_threads(2)
    check_data()
    if args.data_only:return
    root=Path(args.out);rows=json.loads((root/'metrics.json').read_text())
    for r in rows:
        seed,v=r['seed'],r['variant'];torch.manual_seed(seed);model=build(v);initial={k:t.clone() for k,t in model.state_dict().items()}
        ckpt=torch.load(root/f'{v}_seed{seed}.pt',map_location='cpu',weights_only=True);model.load_state_dict(ckpt['state_dict']);model.eval();assert_frozen(model,initial)
        for i,(cond,cfg) in enumerate(CONDITIONS.items()):
            acc=evaluate(model,dataset(4096,seed*1000+10+i,**cfg))
            assert acc==r[cond],(v,seed,cond,acc,r[cond])
    print(f'PASS: {len(rows)} checkpoints, {len(rows)*len(CONDITIONS)} test accuracies reproduced exactly; frozen tensors unchanged.')
if __name__=='__main__':main()

"""大网+双层调制最小实验；CPU 可运行，无外部数据。"""
import argparse, copy, csv, json, math, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

VARIANTS = ['readout_only', 'controller_only', 'upper_only', 'lower_only', 'modulated', 'slow_backbone', 'rnn', 'mlp']

def dataset(n, seed, steps=8):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, steps, 4, generator=gen)
    q = torch.randint(4, (n,), generator=gen)
    y = (x[:, :steps//2].sum(1).gather(1, q[:, None])[:, 0] > 0).float()
    return x, F.one_hot(q, 4).float(), y

class FlowNet(nn.Module):
    def __init__(self, variant, width=64):
        super().__init__()
        self.variant, self.width = variant, width
        # 固定随机有向图；谱范数 <= 0.9，不依赖训练标签构造。
        mask = (torch.rand(width, width) < .20).float()
        w = torch.randn(width, width) * mask
        w = w * (.9 / torch.linalg.matrix_norm(w, 2))
        self.register_buffer('mask', mask)
        self.W = nn.Parameter(w, requires_grad=variant == 'slow_backbone')
        self.register_buffer('B', torch.randn(4, width) / 2)
        self.register_buffer('C', torch.randn(4, width) * .15)
        # 上层只看任务及相对时间，不看信号/标签，避免直接旁路求解。
        self.upper = nn.Sequential(nn.Linear(5, 16), nn.Tanh(), nn.Linear(16, 4))
        # 下层看任务和四个块的局部平均状态，控制四组出边。
        self.lower = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 4))
        self.head = nn.Linear(width, 1)
        with torch.no_grad():
            self.head.weight.mul_(5)
        self.use_upper = variant in ['controller_only', 'upper_only', 'modulated', 'slow_backbone']
        self.use_lower = variant in ['controller_only', 'lower_only', 'modulated', 'slow_backbone']
        for p in self.upper.parameters(): p.requires_grad_(self.use_upper)
        for p in self.lower.parameters(): p.requires_grad_(self.use_lower)
        for p in self.head.parameters(): p.requires_grad_(variant != 'controller_only')

    def forward(self, x, q, force=None, trace=False):
        h = x.new_zeros(x.size(0), self.width)
        ps, gs = [], []
        for t in range(x.size(1)):
            clock = x.new_full((x.size(0), 1), t / (x.size(1)-1))
            p = torch.sigmoid(self.upper(torch.cat([q, clock], 1))) if self.use_upper else torch.ones_like(q)
            local = h.reshape(-1, 4, self.width//4).mean(-1)
            g = torch.sigmoid(self.lower(torch.cat([q, local], 1))) if self.use_lower else torch.ones_like(q)
            if force == 'upper_open': p = torch.ones_like(p)
            if force == 'lower_open': g = torch.ones_like(g)
            # 闭阀缩小出边；稠密计算实现，不宣称真实稀疏加速。
            h = .8*h + .2*torch.tanh((h*g.repeat_interleave(self.width//4, 1)) @ (self.W*self.mask) + (x[:, t]*p) @ self.B + q @ self.C)
            if trace: ps.append(p.detach()); gs.append(g.detach())
        logits = self.head(h).squeeze(-1)
        if trace: return logits, torch.stack(ps, 1), torch.stack(gs, 1)
        return logits

class Baseline(nn.Module):
    def __init__(self, kind):
        super().__init__(); self.kind = kind
        if kind == 'rnn':
            self.rnn = nn.RNN(8, 64, batch_first=True)
            self.head = nn.Linear(64, 1)
        else:
            # 571 参数，与双层控制器+读出 441 参数大致接近。
            self.net = nn.Sequential(nn.Linear(36, 15), nn.Tanh(), nn.Linear(15, 1))
    def forward(self, x, q):
        if self.kind == 'mlp': return self.net(torch.cat([x.flatten(1), q], 1)).squeeze(-1)
        h, _ = self.rnn(torch.cat([x, q[:, None].expand(-1, x.size(1), -1)], -1))
        return self.head(h[:, -1]).squeeze(-1)

@torch.no_grad()
def accuracy(model, data, force=None):
    x, q, y = data
    logits = model(x, q, force=force) if isinstance(model, FlowNet) else model(x, q)
    return ((logits > 0) == y.bool()).float().mean().item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--seeds', type=int, nargs='+', default=[11, 22, 33])
    ap.add_argument('--out', default='results')
    ap.add_argument('--variants', nargs='+', default=VARIANTS, choices=VARIANTS)
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows, histories = [], {}
    config = vars(args) | {'torch': torch.__version__, 'numpy': np.__version__, 'train_n':2048, 'val_n':512, 'test_n':4096, 'batch':256, 'lr':.003, 'backbone_lr':.00015}
    (out/'config.json').write_text(json.dumps(config, indent=2))
    for seed in args.seeds:
        train, val, test = [dataset(n, seed*100+k) for k,n in enumerate([2048,512,4096])]
        for variant in args.variants:
            torch.manual_seed(seed)
            model = Baseline(variant) if variant in ['rnn', 'mlp'] else FlowNet(variant)
            initial = {k:v.clone() for k,v in model.state_dict().items()}
            params = [p for n,p in model.named_parameters() if p.requires_grad and n != 'W']
            groups = [{'params':params, 'lr':.003}]
            if variant == 'slow_backbone': groups.append({'params':[model.W], 'lr':.00015})
            opt = torch.optim.Adam(groups)
            gen = torch.Generator().manual_seed(seed+5000)
            best_acc, best, history = -1, None, []
            start = time.perf_counter()
            for epoch in range(args.epochs):
                model.train(); order = torch.randperm(len(train[0]), generator=gen); total=0
                for ids in order.split(256):
                    opt.zero_grad(set_to_none=True)
                    loss = F.binary_cross_entropy_with_logits(model(train[0][ids], train[1][ids]), train[2][ids])
                    loss.backward(); nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
                    opt.step(); total += loss.item()*len(ids)
                model.eval(); va = accuracy(model, val)
                history.append({'epoch':epoch+1,'loss':total/len(order),'val_accuracy':va})
                if va > best_acc: best_acc, best = va, copy.deepcopy(model.state_dict())
            elapsed = time.perf_counter()-start
            model.load_state_dict(best); model.eval()
            row = {'seed':seed, 'variant':variant, 'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad), 'test_accuracy':accuracy(model,test), 'val_accuracy':best_acc, 'train_seconds':elapsed}
            if isinstance(model, FlowNet):
                row['backbone_max_change'] = (model.W.detach()-initial['W']).abs().max().item()
                if variant != 'slow_backbone': assert torch.equal(model.W, initial['W'])
                assert torch.equal(model.B, initial['B']) and torch.equal(model.C, initial['C'])
                if variant == 'controller_only':
                    assert torch.equal(model.head.weight,initial['head.weight']) and torch.equal(model.head.bias,initial['head.bias'])
                with torch.no_grad(): _, p, g = model(test[0],test[1],trace=True)
                row['mean_injection'] = p.mean().item(); row['mean_gate'] = g.mean().item()
                selected = p.gather(2,test[1].argmax(1)[:,None,None].expand(-1,8,1)).squeeze(-1)
                row['selected_early_injection'] = selected[:,:4].mean().item()
                row['selected_late_injection'] = selected[:,4:].mean().item()
                if variant == 'modulated':
                    row['force_upper_open_accuracy'] = accuracy(model,test,'upper_open')
                    row['force_lower_open_accuracy'] = accuracy(model,test,'lower_open')
                    np.savez(out/f'trace_seed{seed}.npz', injection=p.mean(0).numpy(), gates=g.mean(0).numpy(), selected_injection=selected.mean(0).numpy())
            rows.append(row); histories[f'{variant}_{seed}'] = history
            torch.save({'variant':variant,'seed':seed,'state_dict':best},out/f'{variant}_seed{seed}.pt')
            (out/'metrics.json').write_text(json.dumps(rows,indent=2))
            (out/'history.json').write_text(json.dumps(histories))
            print(json.dumps(row),flush=True)
    summary = {}
    for v in args.variants:
        rr = [r for r in rows if r['variant']==v]
        summary[v] = {'accuracy_mean':float(np.mean([r['test_accuracy'] for r in rr])), 'accuracy_std':float(np.std([r['test_accuracy'] for r in rr],ddof=1)) if len(rr)>1 else 0., 'trainable':rr[0]['trainable'], 'train_seconds_mean':float(np.mean([r['train_seconds'] for r in rr]))}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with (out/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    print(json.dumps(summary,indent=2))

if __name__ == '__main__': main()

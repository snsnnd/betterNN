"""第八轮：Top-K机制归因。N=256；dense、固定随机Top-K、可学习Top-K严格配对。"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, copy, json, math, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).parent
torch.set_num_threads(int(os.environ.get('FLOW_THREADS', '1')))
torch.use_deterministic_algorithms(True)
N = 256
K = 16
SEEDS = [11, 22, 33, 44, 55]
MODES = ['dense', 'fixed_topk', 'learned_topk']
EPOCHS = 30
THRESHOLD = 0.9
DEVICE = torch.device('cpu')


def data(seed, n, task, steps=20):
    g = torch.Generator().manual_seed(seed)
    ab = torch.randn(n, 2, generator=g)
    r = torch.randint(2, (n,), generator=g)
    x = torch.zeros(n, steps, 2)
    x[:, 0, 0] = ab[:, 0]
    x[:, 1, 1] = ab[:, 1]
    meta = torch.zeros(n, steps, 7)
    meta[:, :, 2] = torch.linspace(0, 1, steps)
    meta[:, 4:, 0] = 1
    meta[:, 4:, 1] = (2 * r.float() - 1)[:, None]
    meta[:, :, 3 + task] = 1
    idx = torch.tensor([[[0, 1], [1, 0]], [[0, 0], [1, 1]], [[1, 1], [0, 0]], [[1, 0], [0, 1]]])[task, r]
    return x, meta, (ab.gather(1, idx) > 0).float()


class Net(nn.Module):
    def __init__(self, n, mode):
        super().__init__()
        self.n = n
        self.m = n // 4
        self.mode = mode
        self.k = K
        self.window = 5
        allowed = torch.tensor([[1, 0, 1, 1], [0, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1.]], dtype=torch.float32)
        mask = (torch.rand(n, n) < .3).float() * allowed.repeat_interleave(self.m, 0).repeat_interleave(self.m, 1)
        w = torch.randn(n, n) * mask
        self.register_buffer('mask', mask)
        self.W = nn.Parameter(.9 * w / torch.linalg.matrix_norm(w, 2))
        b = torch.zeros(2, n)
        b[0, :self.m] = torch.randn(self.m) * .7
        b[1, self.m:2 * self.m] = torch.randn(self.m) * .7
        self.register_buffer('B', b)
        self.write = nn.Sequential(nn.Linear(7, 8), nn.Tanh(), nn.Linear(8, 2))
        self.hold = nn.Linear(7, 4)
        nn.init.zeros_(self.hold.weight)
        nn.init.zeros_(self.hold.bias)
        self.route = nn.Sequential(nn.Linear(11, 16), nn.Tanh(), nn.Linear(16, 16))
        self.headX = nn.Linear(self.m, 1)
        self.headY = nn.Linear(self.m, 1)
        with torch.no_grad():
            self.headX.weight.mul_(5)
            self.headY.weight.mul_(5)
        self.selector = nn.Parameter(torch.randn(4, 4, self.m) * .01, requires_grad=mode == 'learned_topk')

    def masks(self, task):
        z = self.selector[task]
        if self.mode == 'dense':
            return torch.ones_like(z), torch.ones_like(z)
        hard = torch.zeros_like(z).scatter_(-1, z.topk(self.k, dim=-1).indices, 1.)
        if self.mode == 'fixed_topk':
            return hard, hard
        soft = self.k * torch.softmax(z, -1)
        return hard + (soft - soft.detach()), hard

    def forward(self, x, meta, trace=False):
        task = int(meta[0, 0, 3:].argmax())
        assert bool((meta[:, 0, 3:].argmax(-1) == task).all())
        sel, hard = self.masks(task)
        topk = self.mode != 'dense'
        scale = math.sqrt(self.n / (4 * self.k)) if topk else 1.
        active = self.k if topk else self.m
        h = x.new_zeros(len(x), 4, self.m)
        w = (self.W * self.mask).reshape(4, self.m, 4, self.m)
        gs = []
        hs = []
        for t in range(x.shape[1]):
            if self.training and t and t % self.window == 0:
                h = h.detach()
            ctl = meta[:, t]
            p = self.write(ctl).sigmoid()
            a = .2 * self.hold(ctl).sigmoid()
            pooled = h.sum(-1) / active
            gate = self.route(torch.cat([ctl, pooled], 1)).sigmoid().reshape(-1, 4, 4)
            rec = (torch.einsum('bsi,sidj->bsdj', h, w) * gate[:, :, :, None]).sum(1) * scale
            cand = torch.tanh(rec + ((x[:, t] * p) @ self.B).reshape(-1, 4, self.m))
            h = ((1 - a[:, :, None]) * h + a[:, :, None] * cand) * sel[None]
            if trace:
                gs.append(gate.detach())
                hs.append(h.detach())
        out = torch.cat([F.linear(h[:, 2] * scale, self.headX.weight, self.headX.bias),
                         F.linear(h[:, 3] * scale, self.headY.weight, self.headY.bias)], 1)
        return (out, torch.stack(gs, 1), torch.stack(hs, 1), hard.detach()) if trace else out


@torch.no_grad()
def evaluate(m, d):
    m.eval()
    correct = 0
    for ids in torch.arange(len(d[0])).split(128):
        x = d[0][ids].to(DEVICE)
        meta = d[1][ids].to(DEVICE)
        y = d[2][ids].to(DEVICE)
        correct += int(((m(x, meta) > 0) == y.bool()).all(1).sum())
    return correct / len(d[0])


@torch.no_grad()
def diagnostics(m, seed):
    m.eval()
    acts, gates, supports = [], [], []
    counts = m.mask.reshape(4, m.m, 4, m.m).sum((1, 3))
    exists = counts > 0
    for task in range(4):
        x, meta, _ = data(seed * 10000 + 999, 128, task)
        x = x.to(DEVICE)
        meta = meta.to(DEVICE)
        _, g, h, hard = m(x, meta, trace=True)
        assert int(hard.sum()) == (m.n if m.mode == 'dense' else 4 * m.k)
        if m.mode != 'dense':
            assert int((h != 0).sum((-1, -2)).max()) <= 4 * m.k
        acts.append(h[:, 4:].abs().mean((0, 1)).flatten())
        gates.append(g[:, 4:].mean((0, 1))[exists])
        supports.append(hard.flatten().bool())
    a = torch.stack(acts)
    gate = torch.stack(gates)
    support = torch.stack(supports)
    kk = max(1, m.n // 4)
    active = torch.zeros_like(a, dtype=torch.bool).scatter_(1, a.topk(kk, dim=1).indices, True)
    def jac(q):
        return [[float((u & v).sum() / (u | v).sum().clamp_min(1)) for v in q] for u in q]
    mx = a.max(0).values
    other = (a.sum(0) - mx) / 3
    si = (mx - other) / (mx + other + 1e-9)
    cosine = F.normalize(gate, dim=1) @ F.normalize(gate, dim=1).T
    return {'activation_top25pct_jaccard': jac(active), 'selection_jaccard': jac(support),
            'node_selectivity_mean': float(si.mean()), 'route_cosine': cosine.tolist(),
            'mean_activation': a.tolist(), 'mean_route': gate.tolist(),
            'selected_nodes': support.nonzero().tolist(),
            'nonzero_activation_nodes': int((a.sum(0) > 1e-9).sum())}


def run(n, mode, seed, out, epochs):
    name = f'{mode}_{n}_{seed}'
    p = out / (name + '.json')
    if p.exists():
        return
    torch.manual_seed(seed)
    m = Net(n, mode).to(DEVICE)
    selector_init = copy.deepcopy(m.selector.detach())
    fixed = {k: v.clone() for k, v in m.state_dict().items() if k in ['B', 'mask']}
    tests = [data(seed * 10000 + t * 100 + 10, 1024, t) for t in range(4)]
    matrix = []
    history = {}
    diag0 = diagnostics(m, seed)
    start = time.perf_counter()
    for task in range(4):
        tr = data(seed * 10000 + task * 100, 512, task)
        val = data(seed * 10000 + task * 100 + 1, 256, task)
        opt = torch.optim.Adam([q for q in m.parameters() if q.requires_grad], lr=.003)
        gen = torch.Generator().manual_seed(seed + task * 100 + 5000)
        hh = []
        for epoch in range(epochs):
            m.train()
            loss_sum = 0
            for ids in torch.randperm(512, generator=gen).split(128):
                xb = tr[0][ids].to(DEVICE)
                mb = tr[1][ids].to(DEVICE)
                yb = tr[2][ids].to(DEVICE)
                opt.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(m(xb, mb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 1.)
                opt.step()
                loss_sum += float(loss.detach()) / 4
            hh.append({'loss': loss_sum, 'validation': evaluate(m, val)})
        history[str(task)] = hh
        matrix.append([evaluate(m, d) for d in tests])
        for k, v in fixed.items():
            assert torch.equal(v.cpu(), m.state_dict()[k].cpu())
        print(name, 'stage', task, [round(v, 3) for v in matrix[-1]], flush=True)
    diag = diagnostics(m, seed)
    ious = [len(set(u) & set(v)) / max(1, len(set(u) | set(v)))
            for u, v in zip(diag0['selected_nodes'], diag['selected_nodes'])]
    row = {'threads': torch.get_num_threads(), 'device': str(DEVICE), 'torch': torch.__version__,
           'name': name, 'n': n, 'k_per_role': K, 'mode': mode, 'seed': seed,
           'active_budget': n if mode == 'dense' else 4 * K, 'matrix': matrix,
           'plasticity': float(np.mean([matrix[t][t] for t in range(4)])),
           'final_mean': float(np.mean(matrix[-1])),
           'forgetting': float(np.mean([max(matrix[s][t] for s in range(t, 4)) - matrix[-1][t] for t in range(3)])),
           'bwt': float(np.mean([matrix[-1][t] - matrix[t][t] for t in range(3)])),
           'epochs_to_90': {str(t): next((i + 1 for i, h in enumerate(hh) if h['validation'] >= THRESHOLD), None)
                            for t, hh in history.items()},
           'support_iou_init_final': float(np.mean(ious)),
           'selector_l2_change': float((m.selector.detach() - selector_init).norm().item()),
           'diag_initial': diag0, 'diag_final': diag, 'history': history,
           'seconds': time.perf_counter() - start,
           'trainable': sum(q.numel() for q in m.parameters() if q.requires_grad)}
    torch.save({'state': m.state_dict(), 'n': n, 'mode': mode, 'seed': seed}, out / (name + '.pt'))
    p.write_text(json.dumps(row, indent=2))


def pick_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=N)
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--modes', nargs='+', choices=MODES, default=MODES)
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--out', default='results')
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()
    global DEVICE
    DEVICE = pick_device(args.device)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(
        {'n': N, 'k_per_role': K, 'active_budget': 4 * K, 'seeds': SEEDS, 'modes': MODES,
         'epochs_per_task': EPOCHS, 'tasks': 4, 'train': 512, 'val': 256, 'test': 1024, 'batch': 128,
         'bptt_window': 5, 'lr': .003, 'selection': 'final epoch', 'replay': False,
         'task_identity': 'visible from t0', 'rule': 'visible from t4',
         'torch': torch.__version__, 'device': str(DEVICE)}, indent=2))
    print('device', DEVICE, 'torch', torch.__version__, flush=True)
    for seed in args.seeds:
        for mode in args.modes:
            run(args.n, mode, seed, out, args.epochs)


if __name__ == '__main__':
    main()

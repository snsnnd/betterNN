"""第十六轮：Sample vs Route/Hold Policy replay（固定带宽，扫描存储字节）。"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).parent
torch.set_num_threads(int(os.environ.get('FLOW_THREADS', '1')))
torch.use_deterministic_algorithms(True)
N = 256
SEEDS = [11, 22, 33, 44, 55]
EPOCHS = 65
LR = .003
BATCH = 128
PER_STEP = 16           # 每步旧信息带宽（样本或锚点）
LAMBDA = 0.5            # policy 蒸馏权重
THRESHOLD = .9
SAMPLE_BYTES = 182 * 4
ROUTE_BYTES = (11 + 16) * 4
HOLD_BYTES = (7 + 4) * 4
RH_BYTES = (11 + 16 + 4) * 4
ORDERS = {'o0': [0, 1, 2, 3], 'o1': [3, 2, 1, 0], 'o2': [1, 3, 0, 2]}
BUDGETS = {
    'none': [0],
    'sample': [1, 2, 4, 8, 16, 32],
    'route': [7, 28, 112],
    'hold': [17, 68, 272],
    'route_hold': [6, 12, 24, 48, 96, 192],
    'hybrid': [(24, 1), (96, 2), (192, 4)],
}
METHODS = list(BUDGETS)
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
    idx = torch.tensor([[[0, 1], [1, 0]], [[0, 0], [1, 1]],
                        [[0, 1], [0, 0]], [[0, 1], [1, 1]]])[task, r]
    return x, meta, (ab.gather(1, idx) > 0).float()


class FlowV2(nn.Module):
    def __init__(self):
        super().__init__()
        n = N
        self.n = n
        self.m = n // 4
        allowed = torch.tensor([[1, 0, 1, 1], [0, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1.]], dtype=torch.float32)
        mask = (torch.rand(n, n) < .3).float() * allowed.repeat_interleave(self.m, 0).repeat_interleave(self.m, 1)
        w = torch.randn(n, n) * mask
        self.register_buffer('mask', mask)
        self.W = nn.Parameter(.9 * w / torch.linalg.matrix_norm(w, 2))
        b = torch.zeros(2, n)
        b[0, :self.m] = torch.randn(self.m) * .7
        b[1, self.m:2 * self.m] = torch.randn(self.m) * .7
        self.register_buffer('B', b)
        _rng_pad = nn.Sequential(nn.Linear(7, 8), nn.Tanh(), nn.Linear(8, 2))
        del _rng_pad
        self.hold = nn.Linear(7, 4)
        nn.init.zeros_(self.hold.weight)
        nn.init.zeros_(self.hold.bias)
        self.route = nn.Sequential(nn.Linear(11, 16), nn.Tanh(), nn.Linear(16, 16))
        self.headX = nn.Linear(self.m, 1)
        self.headY = nn.Linear(self.m, 1)
        with torch.no_grad():
            self.headX.weight.mul_(5)
            self.headY.weight.mul_(5)

    def forward(self, x, meta, record=False):
        h = x.new_zeros(len(x), 4, self.m)
        w = (self.W * self.mask).reshape(4, self.m, 4, self.m)
        rec = {'q': [], 'rlog': [], 'hlog': []} if record else None
        for t in range(x.shape[1]):
            if self.training and t and t % 5 == 0:
                h = h.detach()
            ctl = meta[:, t]
            hlog = self.hold(ctl)
            a = .2 * hlog.sigmoid()
            pooled = h.sum(-1) / self.m
            q = torch.cat([ctl, pooled], 1)
            rlog = self.route(q)
            if record:
                rec['q'].append(q.detach())
                rec['rlog'].append(rlog.detach())
                rec['hlog'].append(hlog.detach())
            gate = rlog.sigmoid().reshape(-1, 4, 4)
            r = (torch.einsum('bsi,sidj->bsdj', h, w) * gate[:, :, :, None]).sum(1)
            cand = torch.tanh(r + (x[:, t] @ self.B).reshape(-1, 4, self.m))
            h = (1 - a[:, :, None]) * h + a[:, :, None] * cand
        out = torch.cat([F.linear(h[:, 2], self.headX.weight, self.headX.bias),
                         F.linear(h[:, 3], self.headY.weight, self.headY.bias)], 1)
        return (out, rec) if record else out


def build(seed):
    torch.manual_seed(seed)
    return FlowV2().to(DEVICE)


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


def scores(m, seed):
    return [evaluate(m, data(seed * 10000 + t * 100 + 10, 1024, t)) for t in range(4)]


@torch.no_grad()
def collect_anchors(m, seed, task, count, gen):
    m.eval()
    tr = data(seed * 10000 + task * 100, 512, task)
    qs, rs, hs = [], [], []
    for ids in torch.arange(512).split(256):
        _, rec = m(tr[0][ids], tr[1][ids], record=True)
        qs.append(torch.stack(rec['q'], 0).reshape(-1, 11))
        rs.append(torch.stack(rec['rlog'], 0).reshape(-1, 16))
        hs.append(torch.stack(rec['hlog'], 0).reshape(-1, 4))
    q = torch.cat(qs); r = torch.cat(rs); h = torch.cat(hs)
    idx = torch.randperm(len(q), generator=gen)[:count]
    return q[idx], r[idx], h[idx]


def train_stage(m, seed, task, epochs, lr, method, samples, anchors, gen_anc):
    tr = data(seed * 10000 + task * 100, 512, task)
    val = data(seed * 10000 + task * 100 + 1, 256, task)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=lr)
    gen = torch.Generator().manual_seed(seed + task * 100 + 5000)
    gen_rep = torch.Generator().manual_seed(seed + task * 100 + 9000)
    use_samples = method in ('sample', 'hybrid') and samples is not None and len(samples[0]) > 0
    use_policy = method in ('route', 'hold', 'route_hold', 'hybrid') and anchors is not None and len(anchors['q']) > 0
    use_route = method in ('route', 'route_hold', 'hybrid')
    use_hold = method in ('hold', 'route_hold', 'hybrid')
    replay_n = PER_STEP if use_samples else 0
    cur_n = BATCH - replay_n
    hh = []
    for epoch in range(epochs):
        m.train()
        loss_sum = 0
        order = torch.randperm(512, generator=gen)
        chunks = order[:4 * cur_n].split(cur_n)
        for chunk in chunks:
            if use_samples:
                bx, bm, by = (torch.cat(b) for b in samples)
                rid = torch.randint(len(bx), (replay_n,), generator=gen_rep)
                xb = torch.cat([tr[0][chunk], bx[rid]]).to(DEVICE)
                mb = torch.cat([tr[1][chunk], bm[rid]]).to(DEVICE)
                yb = torch.cat([tr[2][chunk], by[rid]]).to(DEVICE)
            else:
                xb = tr[0][chunk].to(DEVICE)
                mb = tr[1][chunk].to(DEVICE)
                yb = tr[2][chunk].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(m(xb, mb), yb)
            if use_policy:
                k = min(PER_STEP, len(anchors['q']))
                aid = torch.randint(len(anchors['q']), (k,), generator=gen_anc)
                if use_route:
                    loss = loss + LAMBDA * F.mse_loss(m.route(anchors['q'][aid]), anchors['rlog'][aid])
                if use_hold:
                    loss = loss + LAMBDA * F.mse_loss(m.hold(anchors['q'][aid][:, :7]), anchors['hlog'][aid])
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.)
            opt.step()
            loss_sum += float(loss.detach()) / len(chunks)
        hh.append({'loss': loss_sum, 'validation': evaluate(m, val)})
    return hh


def metrics(matrix, order):
    n = len(order)
    acq = float(np.mean([matrix[i][order[i]] for i in range(n)]))
    final = float(np.mean(matrix[-1]))
    forget = float(np.mean([max(matrix[s][t] for s in range(i, n)) - matrix[-1][t] for i, t in enumerate(order[:-1])]))
    bwt = float(np.mean([matrix[-1][t] - matrix[i][t] for i, t in enumerate(order[:-1])]))
    return acq, final, forget, bwt


def drift(m, pre_state):
    out = {}
    named = list(m.named_parameters())
    for grp, keys in [('W', ['W']), ('route', None), ('hold', None), ('readout', ['headX', 'headY'])]:
        tot = 0.0
        for n, p in named:
            if grp == 'route' and not n.startswith('route'):
                continue
            if grp == 'hold' and not n.startswith('hold'):
                continue
            if grp == 'W' and n != 'W':
                continue
            if grp == 'readout' and n.split('.')[0] not in ('headX', 'headY'):
                continue
            tot += float((p.detach().cpu() - pre_state[n]).pow(2).sum())
        out['drift_' + grp] = tot ** .5
    return out


def label_for(method, budget):
    if method == 'none':
        return 'none', {'anchors': 0, 'samples': 0, 'bytes': 0}
    if method == 'sample':
        return f's{budget}', {'anchors': 0, 'samples': budget, 'bytes': budget * SAMPLE_BYTES}
    if method == 'route':
        return f'r{budget}', {'anchors': budget, 'samples': 0, 'bytes': budget * ROUTE_BYTES}
    if method == 'hold':
        return f'h{budget}', {'anchors': budget, 'samples': 0, 'bytes': budget * HOLD_BYTES}
    if method == 'route_hold':
        return f'rh{budget}', {'anchors': budget, 'samples': 0, 'bytes': budget * RH_BYTES}
    a, s = budget
    return f'rh{a}s{s}', {'anchors': a, 'samples': s, 'bytes': a * RH_BYTES + s * SAMPLE_BYTES}


def run_stream(method, budget, order_key, seed, out, epochs):
    label, storage = label_for(method, budget)
    name = f'{order_key}_{method}_{label}_{seed}'
    p = out / (name + '.json')
    if p.exists():
        return
    order = ORDERS[order_key]
    m = build(seed)
    samples = ([], [], [])
    anchors = {'q': torch.zeros(0, 11), 'rlog': torch.zeros(0, 16), 'hlog': torch.zeros(0, 4)}
    gen_anc = torch.Generator().manual_seed(seed + 7000)
    matrix, history, epochs_to_90 = [], {}, {}
    start = time.perf_counter()
    for i, task in enumerate(order):
        hh = train_stage(m, seed, task, epochs, LR, method, samples, anchors, gen_anc)
        history[str(task)] = hh
        epochs_to_90[str(task)] = next((j + 1 for j, h in enumerate(hh) if h['validation'] >= THRESHOLD), None)
        matrix.append(scores(m, seed))
        torch.save({'state': m.state_dict(), 'method': method, 'label': label, 'order': order_key,
                    'seed': seed, 'stage': i, 'task': task}, out / f'{name}_stage{i}.pt')
        if storage['samples'] > 0:
            tr = data(seed * 10000 + task * 100, 512, task)
            for k in range(3):
                samples[k].append(tr[k][:storage['samples']])
        if storage['anchors'] > 0:
            q, r, h = collect_anchors(m, seed, task, storage['anchors'], gen_anc)
            anchors['q'] = torch.cat([anchors['q'], q])
            anchors['rlog'] = torch.cat([anchors['rlog'], r])
            anchors['hlog'] = torch.cat([anchors['hlog'], h])
        print(name, 'stage', i, [round(v, 3) for v in matrix[-1]], flush=True)
    pre_state = torch.load(out / f'{name}_stage0.pt', map_location='cpu', weights_only=False)['state']
    acq, final, forget, bwt = metrics(matrix, order)
    row = {'name': name, 'method': method, 'label': label, 'order': order_key, 'order_seq': order, 'seed': seed,
           'matrix': matrix, 'history': history, 'epochs_to_90': epochs_to_90,
           'acquisition': acq, 'final_mean': final, 'forgetting': forget, 'bwt': bwt,
           'anchors': storage['anchors'], 'samples': storage['samples'], 'bytes_per_task': storage['bytes'],
           'bands_per_step': PER_STEP, 'lambda': LAMBDA,
           'seconds': time.perf_counter() - start, 'threads': torch.get_num_threads(),
           'device': str(DEVICE), 'torch': torch.__version__, 'epochs': epochs, 'lr': LR, **drift(m, pre_state)}
    p.write_text(json.dumps(row, indent=2))


def pick_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def main():
    global DEVICE, LAMBDA
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    ap.add_argument('--orders', nargs='+', default=list(ORDERS))
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--limit-budgets', type=int, default=0)
    ap.add_argument('--labels', nargs='+', default=None)
    ap.add_argument('--policy-lambda', type=float, default=LAMBDA)
    ap.add_argument('--out', default='results')
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()
    DEVICE = pick_device(args.device)
    LAMBDA = args.policy_lambda
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(
        {'n': N, 'seeds': SEEDS, 'orders': ORDERS, 'methods': METHODS, 'budgets': BUDGETS,
         'band_per_step': PER_STEP, 'lambda': LAMBDA, 'sample_bytes': SAMPLE_BYTES, 'route_bytes': ROUTE_BYTES,
         'hold_bytes': HOLD_BYTES, 'rh_bytes': RH_BYTES, 'epochs': args.epochs, 'lr': LR, 'batch': BATCH,
         'torch': torch.__version__, 'device': str(DEVICE)}, indent=2))
    print('device', DEVICE, 'methods', args.methods, 'orders', args.orders, 'epochs', args.epochs, flush=True)
    for method in args.methods:
        budgets = BUDGETS[method]
        if args.limit_budgets:
            budgets = budgets[:args.limit_budgets]
        for order_key in args.orders:
            for seed in args.seeds:
                for budget in budgets:
                    label, _ = label_for(method, budget)
                    if args.labels and label not in args.labels:
                        continue
                    run_stream(method, budget, order_key, seed, out, args.epochs)


if __name__ == '__main__':
    main()

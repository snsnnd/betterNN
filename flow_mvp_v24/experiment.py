"""第二十四轮：Input Write Dynamics。

- 写入 trace：z_t = Σ_k a_k B x_{t-k}（因果 FIR，Σa²=1）；
- Phase delay：固定 B（V20 fixed-disjoint）、V20 训练协议，扫 kernel × T{20,40,80,160}；
- Phase chain：V23 Chain-select + learnable B hybrid（core K=5 / B full），只改 kernel。
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).parent
torch.set_num_threads(int(os.environ.get('FLOW_THREADS', '1')))
torch.use_deterministic_algorithms(True)

N = 256
M = N // 4
WRITE = 2 * M
BETA = .7 * math.sqrt(M)
SEEDS = [11, 22, 33, 44, 55]
EPOCHS = 65
CHAIN_EPOCHS = 500
LR = .003
BATCH = 128
KERNELS = ['single', 'burst3', 'burst5', 'decay-fast', 'decay-slow']
CHAIN_KERNELS = ['single', 'burst5', 'decay-slow']
ORDERS = {'o0': [0, 1, 2, 3], 'o1': [3, 2, 1, 0], 'o2': [1, 3, 0, 2]}


def make_kernel(name):
    if name == 'single':
        return torch.tensor([1.0])
    if name == 'burst3':
        return torch.full((3,), 1 / math.sqrt(3))
    if name == 'burst5':
        return torch.full((5,), 1 / math.sqrt(5))
    if name == 'decay-fast':
        a = torch.tensor([0.5 ** k for k in range(5)])
        return a / a.norm()
    if name == 'decay-slow':
        a = torch.tensor([0.85 ** k for k in range(5)])
        return a / a.norm()
    raise ValueError(name)


def data(seed, n, task, steps=20):
    """V20 A'B'C'D' 延迟任务。"""
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


def data_chain(seed, n, task, T=80):
    """V23 Chain-select（瞬态脉冲 + 条件选择）。"""
    g = torch.Generator().manual_seed(seed)
    A = torch.where(torch.rand(n, generator=g) < .5, -1., 1.)
    Bv = torch.where(torch.rand(n, generator=g) < .5, -1., 1.)
    c1 = torch.randint(0, 2, (n,), generator=g).float() * 4 - 2
    c2 = torch.randint(0, 2, (n,), generator=g).float() * 4 - 2
    d = torch.randn(n, generator=g)
    tauA, tauC1, tauD, tauB, tauC2 = [max(1, min(T - 2, int(round(f * T)))) for f in (.05, .21, .39, .58, .76)]
    x = torch.zeros(n, T, 2)
    x[:, tauA, 0] = A
    x[:, tauB, 1] = Bv
    x[:, tauD, 0] = d
    x[:, tauC1, 0] = c1
    x[:, tauC2, 1] = c2
    meta = torch.zeros(n, T, 7)
    meta[:, :, 2] = torch.linspace(0, 1, T)
    meta[:, tauC2:, 3] = 1
    same = (c1 == c2)
    s = lambda v: (v > 0).float()
    if task == 'xor':
        y = torch.stack([torch.where(same, s(A), s(Bv)), (c1 > 0).float()], 1)
    elif task == 'xorsw':
        y = torch.stack([torch.where(same, s(Bv), s(A)), (c2 > 0).float()], 1)
    else:
        raise ValueError(task)
    return x, meta, y


class Flow(nn.Module):
    """V22/V23 Flow + 固定 write kernel；learnable=False 时与 V20 fixed 逐位同构。"""

    def __init__(self, sw=.9, learnable=False, kernel_name='single'):
        super().__init__()
        n = N
        self.n = n
        self.m = n // 4
        self.beta = BETA
        allowed = torch.tensor([[1, 0, 1, 1], [0, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1.]], dtype=torch.float32)
        mask = (torch.rand(n, n) < .3).float() * allowed.repeat_interleave(self.m, 0).repeat_interleave(self.m, 1)
        w = torch.randn(n, n) * mask
        self.register_buffer('mask', mask)
        self.W = nn.Parameter(sw * w / torch.linalg.matrix_norm(w, 2))
        b = torch.zeros(2, n)
        b[0, :self.m] = torch.randn(self.m) * .7
        b[1, self.m:2 * self.m] = torch.randn(self.m) * .7
        self.register_buffer('B', b)
        if learnable:
            self.B_raw = nn.Parameter(b[:, :WRITE].clone())
        else:
            self.B_raw = None
        self.detach_window = True
        self.detach_period = 5
        self.register_buffer('kernel', make_kernel(kernel_name))
        self.kernel_name = kernel_name
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

    def eff_B(self):
        if self.B_raw is None:
            return self.B
        n = self.B_raw.norm(dim=1, keepdim=True) + 1e-12
        be = self.beta * self.B_raw / n
        pad = torch.zeros(2, N - WRITE, device=be.device, dtype=be.dtype)
        return torch.cat([be, pad], dim=1)

    def convolved(self, x):
        if self.kernel.numel() == 1:
            return x
        n, T, _ = x.shape
        xt = x.transpose(1, 2)
        pad = x.new_zeros(n, 2, self.kernel.numel() - 1)
        xt = torch.cat([pad, xt], dim=2)
        w = self.kernel.view(1, 1, -1).expand(2, 1, -1)
        return F.conv1d(xt, w, groups=2).transpose(1, 2)

    def forward(self, x, meta):
        B = self.eff_B()
        xw = self.convolved(x)
        h = x.new_zeros(len(x), 4, self.m)
        w = (self.W * self.mask).reshape(4, self.m, 4, self.m)
        for t in range(x.shape[1]):
            if self.training and self.detach_window and t and t % self.detach_period == 0:
                h = h.detach()
            ctl = meta[:, t]
            a = .2 * self.hold(ctl).sigmoid()
            gate = self.route(torch.cat([ctl, h.sum(-1) / self.m], 1)).sigmoid().reshape(-1, 4, 4)
            rec = (torch.einsum('bsi,sidj->bsdj', h, w) * gate[:, :, :, None]).sum(1)
            cand = torch.tanh(rec + (xw[:, t] @ B).reshape(-1, 4, self.m))
            h = (1 - a[:, :, None]) * h + a[:, :, None] * cand
        return torch.cat([F.linear(h[:, 2], self.headX.weight, self.headX.bias),
                          F.linear(h[:, 3], self.headY.weight, self.headY.bias)], 1)


def build(seed, sw=.9, learnable=False, kernel_name='single'):
    torch.manual_seed(seed)
    return Flow(sw, learnable, kernel_name)


def make_fixed_B(m, seed, alpha=0.0):
    """V20 overlap 构造（含 fixed-disjoint α=0）。"""
    g = torch.Generator().manual_seed(seed * 104729 + 6 * 1009 + int(round(alpha * 10)))
    per_role = 8
    shared = int(round(per_role * alpha))
    shift = per_role - shared
    amp = BETA / math.sqrt(2 * per_role)
    B = torch.zeros(2, N)
    for role in (0, 1):
        pool = torch.randperm(M, generator=g)
        ch0 = pool[:per_role]
        ch1 = pool[shift:shift + per_role]
        s0 = torch.randint(0, 2, (per_role,), generator=g).float() * 2 - 1
        s1 = torch.randint(0, 2, (per_role,), generator=g).float() * 2 - 1
        for i in range(shared):
            s1[i] = s0[shift + i] * (1 if i % 2 == 0 else -1)
        B[0, role * M + ch0] = amp * s0
        B[1, role * M + ch1] = amp * s1
    with torch.no_grad():
        m.B.copy_(B)


@torch.no_grad()
def evaluate(m, d):
    m.eval()
    out = {'acc': 0, 'y0': 0, 'y1': 0}
    for ids in torch.arange(len(d[0])).split(128):
        pred = m(d[0][ids], d[1][ids]) > 0
        y = d[2][ids].bool()
        out['acc'] += int((pred == y).all(1).sum())
        out['y0'] += int((pred[:, 0] == y[:, 0]).sum())
        out['y1'] += int((pred[:, 1] == y[:, 1]).sum())
    n = len(d[0])
    return {k: v / n for k, v in out.items()}


# ---------------- Phase delay：V20 固定 B 协议 ----------------

def train_fixed(m, seed, task, T, epochs):
    tr = data(seed * 10000 + task * 100, 512, task, T)
    val = data(seed * 10000 + task * 100 + 1, 256, task, T)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=LR)
    gen = torch.Generator().manual_seed(seed + task * 100 + 5000)
    curve = []
    for epoch in range(epochs):
        m.train()
        order = torch.randperm(512, generator=gen)
        for ch in order.split(BATCH):
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(m(tr[0][ch], tr[1][ch]), tr[2][ch])
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.)
            opt.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            curve.append({'epoch': epoch + 1, **evaluate(m, val)})
    return curve


@torch.no_grad()
def _traj(m, x, meta, times):
    xw = m.convolved(x)
    B = m.eff_B()
    w = (m.W * m.mask).reshape(4, M, 4, M)
    h = x.new_zeros(len(x), 4, M)
    out = {}
    for t in range(x.shape[1]):
        ctl = meta[:, t]
        a = .2 * m.hold(ctl).sigmoid()
        gate = m.route(torch.cat([ctl, h.sum(-1) / M], 1)).sigmoid().reshape(-1, 4, 4)
        rec = (torch.einsum('bsi,sidj->bsdj', h, w) * gate[:, :, :, None]).sum(1)
        cand = torch.tanh(rec + (xw[:, t] @ B).reshape(-1, 4, M))
        h = (1 - a[:, :, None]) * h + a[:, :, None] * cand
        if t in times:
            out[t] = h.clone()
    return out


@torch.no_grad()
def response_metrics(m, seed, task, T, n=64):
    x, meta, _ = data(seed * 10000 + task * 100 + 10, n, task, T)
    xA = x.clone(); xA[:, :, 1] = 0
    xB = x.clone(); xB[:, :, 0] = 0
    x0 = torch.zeros_like(x)
    slope_ts = list(range(5, T, max(1, T // 16)))
    times = sorted(set([2, 5, T // 4, T // 2, (3 * T) // 4, T - 1]) | set(slope_ts))
    tset = set(times)
    hA = _traj(m, xA, meta, tset)
    hB = _traj(m, xB, meta, tset)
    h0 = _traj(m, x0, meta, tset)
    dhA = {t: float((hA[t] - h0[t]).norm() / math.sqrt(n)) for t in times}
    dhB = {t: float((hB[t] - h0[t]).norm() / math.sqrt(n)) for t in times}
    ys = np.array([dhA[t] for t in slope_ts])
    slope = float(np.polyfit(slope_ts, np.log(ys + 1e-12), 1)[0]) if len(ys) > 1 else float('nan')
    ref = dhA.get(5, float('nan'))
    return {'dhA': {str(k): v for k, v in dhA.items()}, 'dhB': {str(k): v for k, v in dhB.items()},
            'ref5': ref,
            'ret_quarter': dhA.get(T // 4, float('nan')) / ref if ref else float('nan'),
            'ret_half': dhA.get(T // 2, float('nan')) / ref if ref else float('nan'),
            'ret_end': dhA.get(T - 1, float('nan')) / ref if ref else float('nan'),
            'decay_slope': slope, 'lifetime': -1 / slope if slope < 0 else float('inf')}


@torch.no_grad()
def probe_retention(m, seed, task, T, n=1024, times=None):
    """在中间状态上拟合线性 probe（前一半训练、后一半评估），测“早期信息还可解码吗”。"""
    x, meta, y = data(seed * 10000 + task * 100 + 10, n, task, T)
    times = times or sorted({5, T // 2, T - 1})
    h = _traj(m, x, meta, set(times))
    out = {}
    for t in times:
        H = h[t].reshape(n, -1)
        Htr, Hte = H[:512], H[512:]
        ytr, yte = y[:512], y[512:]
        accs = []
        for j in range(2):
            w = torch.linalg.lstsq(Htr, ytr[:, j:j + 1]).solution
            pred = (Hte @ w) > 0.5
            accs.append(float((pred.squeeze(1) == (yte[:, j] > 0.5)).float().mean()))
        out[str(t)] = float(np.mean(accs))
    return out


def run_delay(kernel, T, task, seed, out, epochs=EPOCHS):
    name = f'delay_{kernel}_T{T}_t{task}_{seed}'
    p = out / (name + '.json')
    if p.exists():
        return
    m = build(seed, .9, learnable=False, kernel_name=kernel)
    make_fixed_B(m, seed, 0.0)
    init_ret = response_metrics(m, seed, task, T)
    start = time.perf_counter()
    curve = train_fixed(m, seed, task, T, epochs)
    final = evaluate(m, data(seed * 10000 + task * 100 + 10, 1024, task, T))
    ret = response_metrics(m, seed, task, T)
    probe = probe_retention(m, seed, task, T)
    row = {'name': name, 'phase': 'delay', 'kernel': kernel, 'T': T, 'task': task, 'seed': seed,
           'final': final['acc'], 'y0': final['y0'], 'y1': final['y1'], 'curve': curve,
           'retention': ret, 'retention_init': init_ret, 'probe': probe, 'epochs': epochs,
           'seconds': time.perf_counter() - start, 'lr': LR}
    p.write_text(json.dumps(row, indent=2))
    print(f'{name}: acc={final["acc"]:.3f} ret_half={ret["ret_half"]:.3f} '
          f'probe={ {k: round(v,3) for k,v in probe.items()} } {row["seconds"]:.0f}s', flush=True)


# ---------------- Phase chain：V23 hybrid（core K=5 / B full） ----------------

def train_chain(m, seed, task, T, epochs):
    full_names = {n for n, _ in m.named_parameters() if n == 'B_raw' or n.split('.')[0] in ('headX', 'headY')}
    named = list(m.named_parameters())
    core = [p for n, p in named if n != 'B_raw']
    full_params = [p for n, p in named if n in full_names]
    opt_core = torch.optim.Adam(core, lr=LR)
    opt_b = torch.optim.Adam([m.B_raw], lr=LR)
    tr = data_chain(seed * 10000, 512, task, T)
    val = data_chain(seed * 10000 + 1, 256, task, T)
    gen = torch.Generator().manual_seed(seed + 5000)
    curve = []
    for epoch in range(epochs):
        m.train()
        m.detach_period = 5
        order = torch.randperm(512, generator=gen)
        for ch in order.split(BATCH):
            xb, mb, yb = tr[0][ch], tr[1][ch], tr[2][ch]
            opt_core.zero_grad(set_to_none=True)
            opt_b.zero_grad(set_to_none=True)
            m.detach_window = True
            F.binary_cross_entropy_with_logits(m(xb, mb), yb).backward()
            gt = {n: p.grad.detach().clone() for n, p in named if p.grad is not None}
            for p in m.parameters():
                p.grad = None
            m.eval()
            m.detach_window = False
            lf = F.binary_cross_entropy_with_logits(m(xb, mb), yb)
            gf = torch.autograd.grad(lf, full_params)
            m.train()
            gi = 0
            for n, p in named:
                if n in full_names:
                    p.grad = gf[gi]
                    gi += 1
                else:
                    p.grad = gt[n]
            nn.utils.clip_grad_norm_(core, 1.)
            nn.utils.clip_grad_norm_([m.B_raw], 1.)
            opt_core.step()
            opt_b.step()
        if (epoch + 1) % 25 == 0 or epoch == 0:
            curve.append({'epoch': epoch + 1, **evaluate(m, val)})
    return curve


def run_chain(kernel, task, seed, out, T=80, epochs=CHAIN_EPOCHS):
    name = f'chain_{kernel}_{task}_T{T}_{seed}'
    p = out / (name + '.json')
    if p.exists():
        return
    m = build(seed, .9, learnable=True, kernel_name=kernel)
    with torch.no_grad():
        m.B_raw.copy_(make_fixed_B_raw(seed))
    start = time.perf_counter()
    curve = train_chain(m, seed, task, T, epochs)
    te = data_chain(seed * 10000 + 10, 1024, task, T)
    final = evaluate(m, te)
    best = max(c['acc'] for c in curve) if curve else float('nan')
    row = {'name': name, 'phase': 'chain', 'kernel': kernel, 'task': task, 'T': T, 'seed': seed,
           'final': final['acc'], 'y0': final['y0'], 'y1': final['y1'], 'best_val': best,
           'curve': curve, 'epochs': epochs, 'seconds': time.perf_counter() - start, 'lr': LR}
    p.write_text(json.dumps(row, indent=2))
    print(f'{name}: acc={final["acc"]:.3f} best={best:.3f} {row["seconds"]:.0f}s', flush=True)


def make_fixed_B_raw(seed):
    return make_fixed_B_tensor(seed, 1.0)


def make_fixed_B_tensor(seed, alpha):
    g = torch.Generator().manual_seed(seed * 104729 + 6 * 1009 + int(round(alpha * 10)))
    per_role = 8
    shared = int(round(per_role * alpha))
    shift = per_role - shared
    amp = BETA / math.sqrt(2 * per_role)
    B = torch.zeros(2, N)
    for role in (0, 1):
        pool = torch.randperm(M, generator=g)
        ch0 = pool[:per_role]
        ch1 = pool[shift:shift + per_role]
        s0 = torch.randint(0, 2, (per_role,), generator=g).float() * 2 - 1
        s1 = torch.randint(0, 2, (per_role,), generator=g).float() * 2 - 1
        for i in range(shared):
            s1[i] = s0[shift + i] * (1 if i % 2 == 0 else -1)
        B[0, role * M + ch0] = amp * s0
        B[1, role * M + ch1] = amp * s1
    return B[:, :WRITE].clone()


def _job(args):
    (run_delay if args.pop('kind') == 'delay' else run_chain)(**args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['delay', 'chain'], default='delay')
    ap.add_argument('--kernels', nargs='+', default=None)
    ap.add_argument('--Ts', nargs='+', type=int, default=[20, 40, 80, 160])
    ap.add_argument('--tasks', nargs='+', default=['0', '1', '2', '3'])
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--jobs', type=int, default=1)
    args = ap.parse_args()
    if args.phase == 'delay':
        kernels = args.kernels or KERNELS
        out = ROOT / (args.out or 'results/delay')
        epochs = args.epochs or EPOCHS
        jobs = [dict(kind='delay', kernel=k, T=T, task=int(t), seed=s, out=out, epochs=epochs)
                for k in kernels for T in args.Ts for t in args.tasks for s in args.seeds]
    else:
        kernels = args.kernels or CHAIN_KERNELS
        out = ROOT / (args.out or 'results/chain')
        epochs = args.epochs or CHAIN_EPOCHS
        T = args.Ts[0]
        jobs = [dict(kind='chain', kernel=k, task=t, seed=s, out=out, T=T, epochs=epochs)
                for k in kernels for t in ('xor', 'xorsw') for s in args.seeds]
    out.mkdir(parents=True, exist_ok=True)
    print('phase', args.phase, 'jobs', len(jobs), 'out', out, 'epochs', args.epochs, flush=True)
    t0 = time.perf_counter()
    if args.jobs <= 1:
        for j in jobs:
            _job(j)
    else:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(args.jobs) as pool:
            for _ in pool.imap_unordered(_job, jobs):
                pass
    print('all', len(jobs), 'jobs in', round(time.perf_counter() - t0, 1), 's')


if __name__ == '__main__':
    main()

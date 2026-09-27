"""第二十三轮：Credit-Stress Benchmark（不改模型，只改任务）。

- 任务族 Chain-select：瞬态脉冲 A/c1/distractor/B/c2 + 最后读出；
- B 恒可训练恒 Full；core {W,route,hold} 用 detach_period=K 的截断窗口；
- K<Full 用 V22 双通道（截断通道给 core、完整通道给 B/readout），双 optimizer 分别 clip。
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
EPOCHS = 500
LR = .003
BATCH = 128
TASKS = ['xor', 'xorsw']
VARIANTS = ['transient', 'sustained']
FRACS = (.05, .21, .39, .58, .76)


def group_of(name):
    root = name.split('.')[0]
    if root == 'B_raw':
        return 'B'
    if root == 'W':
        return 'W'
    if root == 'route':
        return 'route'
    if root == 'hold':
        return 'hold'
    if root in ('headX', 'headY'):
        return 'readout'
    raise ValueError(name)


def data_v23(seed, n, task, T, variant='transient'):
    """Chain-select：A@.05 / c1@.21 / distractor@.39 / B@.58 / c2@.76。"""
    g = torch.Generator().manual_seed(seed)
    A = torch.where(torch.rand(n, generator=g) < .5, -1., 1.)
    Bv = torch.where(torch.rand(n, generator=g) < .5, -1., 1.)
    c1 = torch.randint(0, 2, (n,), generator=g).float() * 4 - 2
    c2 = torch.randint(0, 2, (n,), generator=g).float() * 4 - 2
    d = torch.randn(n, generator=g)
    tauA, tauC1, tauD, tauB, tauC2 = [max(1, min(T - 2, int(round(f * T)))) for f in FRACS]
    x = torch.zeros(n, T, 2)
    x[:, tauA, 0] = A
    x[:, tauB, 1] = Bv
    x[:, tauD, 0] = d
    meta = torch.zeros(n, T, 7)
    meta[:, :, 2] = torch.linspace(0, 1, T)
    meta[:, tauC2:, 3] = 1
    if variant == 'transient':
        x[:, tauC1, 0] = c1
        x[:, tauC2, 1] = c2
    else:
        meta[:, tauC1:, 0] = 1
        meta[:, tauC1:, 1] = c1[:, None]
        meta[:, tauC2:, 4] = c2[:, None]
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
    """与 V21/V22 逐位同构的 Flow-v2 + 可训练 B。"""

    def __init__(self, sw=.9, learnable=False, init_raw=None):
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
            raw = b[:, :WRITE].clone() if init_raw is None else init_raw.clone()
            self.B_raw = nn.Parameter(raw)
        else:
            self.B_raw = None
        self.detach_window = True
        self.detach_period = 5
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

    def forward(self, x, meta):
        B = self.eff_B()
        h = x.new_zeros(len(x), 4, self.m)
        w = (self.W * self.mask).reshape(4, self.m, 4, self.m)
        for t in range(x.shape[1]):
            if self.training and self.detach_window and t and t % self.detach_period == 0:
                h = h.detach()
            ctl = meta[:, t]
            a = .2 * self.hold(ctl).sigmoid()
            gate = self.route(torch.cat([ctl, h.sum(-1) / self.m], 1)).sigmoid().reshape(-1, 4, 4)
            rec = (torch.einsum('bsi,sidj->bsdj', h, w) * gate[:, :, :, None]).sum(1)
            cand = torch.tanh(rec + (x[:, t] @ B).reshape(-1, 4, self.m))
            h = (1 - a[:, :, None]) * h + a[:, :, None] * cand
        return torch.cat([F.linear(h[:, 2], self.headX.weight, self.headX.bias),
                          F.linear(h[:, 3], self.headY.weight, self.headY.bias)], 1)


def build(seed, sw=.9, learnable=False, init_raw=None):
    torch.manual_seed(seed)
    return Flow(sw, learnable, init_raw)


def _overlap_B(seed, alpha):
    g = torch.Generator().manual_seed(seed * 104729 + 6 * 1009 + int(round(alpha * 10)))
    per_role, shared = 8, int(round(8 * alpha))
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
    return B


def make_model(seed, sw=.9):
    m = build(seed, sw, learnable=True)
    raw = _overlap_B(seed, 1.0)[:, :WRITE].clone()
    with torch.no_grad():
        m.B_raw.copy_(raw)
    return m


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


def train_single(task, variant, T, K, seed, out, epochs=EPOCHS, lr=LR):
    name = f'{variant}_{task}_T{T}_K{K}_{seed}'
    p = out / (name + '.json')
    if p.exists():
        return
    m = make_model(seed)
    tr = data_v23(seed * 10000, 512, task, T, variant)
    va = data_v23(seed * 10000 + 1, 256, task, T, variant)
    te = data_v23(seed * 10000 + 10, 1024, task, T, variant)
    core = [p_ for n, p_ in m.named_parameters() if n != 'B_raw']
    full_names = {n for n, _ in m.named_parameters() if n == 'B_raw' or group_of(n) == 'readout'}
    full_params = [p_ for n, p_ in m.named_parameters() if n in full_names]
    named = list(m.named_parameters())
    opt_core = torch.optim.Adam(core, lr=lr)
    opt_b = torch.optim.Adam([m.B_raw], lr=lr)
    gen = torch.Generator().manual_seed(seed + 5000)
    curve = []
    start = time.perf_counter()
    for epoch in range(epochs):
        m.train()
        if K != 'full':
            m.detach_period = int(K)
        order = torch.randperm(512, generator=gen)
        for ch in order.split(BATCH):
            xb, mb, yb = tr[0][ch], tr[1][ch], tr[2][ch]
            opt_core.zero_grad(set_to_none=True)
            opt_b.zero_grad(set_to_none=True)
            if K == 'full':
                m.detach_window = False
                loss = F.binary_cross_entropy_with_logits(m(xb, mb), yb)
                loss.backward()
                m.detach_window = True
            else:
                m.detach_window = True
                loss = F.binary_cross_entropy_with_logits(m(xb, mb), yb)
                loss.backward()
                g_trunc = {n: p_.grad.detach().clone() for n, p_ in named if p_.grad is not None}
                for p_ in m.parameters():
                    p_.grad = None
                m.eval()
                m.detach_window = False
                loss_full = F.binary_cross_entropy_with_logits(m(xb, mb), yb)
                g_full = torch.autograd.grad(loss_full, full_params)
                m.train()
                gi = 0
                for n, p_ in named:
                    if n in full_names:
                        p_.grad = g_full[gi]
                        gi += 1
                    else:
                        p_.grad = g_trunc[n]
            nn.utils.clip_grad_norm_(core, 1.)
            nn.utils.clip_grad_norm_([m.B_raw], 1.)
            opt_core.step()
            opt_b.step()
        if (epoch + 1) % 25 == 0 or epoch == 0:
            curve.append({'epoch': epoch + 1, **evaluate(m, va)})
    final = evaluate(m, te)
    row = {'name': name, 'task': task, 'variant': variant, 'T': T, 'K': str(K), 'seed': seed,
           'final': final['acc'], 'y0': final['y0'], 'y1': final['y1'], 'curve': curve,
           'epochs_to_90': next((c['epoch'] for c in curve if c['acc'] >= .9), None),
           'seconds': time.perf_counter() - start, 'epochs': epochs, 'lr': lr,
           'params': sum(p_.numel() for p_ in m.parameters() if p_.requires_grad)}
    p.write_text(json.dumps(row, indent=2))
    print(f'{name}: acc={final["acc"]:.3f} (y0 {final["y0"]:.3f}/y1 {final["y1"]:.3f}) '
          f'{row["seconds"]:.0f}s', flush=True)


def _job(args):
    train_single(**args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='+', default=TASKS)
    ap.add_argument('--variants', nargs='+', default=VARIANTS)
    ap.add_argument('--Ts', nargs='+', type=int, default=[20, 40, 80, 160])
    ap.add_argument('--Ks', nargs='+', default=['5', '10', '20', 'full'])
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--out', default='results/main')
    ap.add_argument('--jobs', type=int, default=1)
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for variant in args.variants:
        for task in args.tasks:
            for T in args.Ts:
                for K in args.Ks:
                    if K != 'full' and int(K) >= T:
                        continue
                    for seed in args.seeds:
                        jobs.append(dict(task=task, variant=variant, T=T, K=K, seed=seed,
                                         out=out, epochs=args.epochs))
    print('jobs', len(jobs), 'out', out, flush=True)
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

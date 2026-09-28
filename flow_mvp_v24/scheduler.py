"""V24B（Phase C）：可学习 Write Scheduler。

w_t = 2σ(MLP(x_t, meta_t))（不看 h）；e_t = w_t ⊙ x_t；z_t = Σ_k a_k (e_{t-k} B)。
controller 末层零初始化 → 初始 w≡1（与固定 kernel 起点逐值一致）。
信用：w 与 B 走 full BPTT（hybrid），core 仍 K=5。
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import experiment as E

ROOT = Path(__file__).parent
torch.set_num_threads(int(os.environ.get('FLOW_THREADS', '1')))
torch.use_deterministic_algorithms(True)

N, M, WRITE = E.N, E.M, E.WRITE


class FlowS(E.Flow):
    """E.Flow + write controller；scheduler=False 时与 E.Flow 逐位同构。"""

    def __init__(self, sw=.9, learnable=False, kernel_name='single', scheduler=False):
        super().__init__(sw, learnable, kernel_name)
        self.scheduler = scheduler
        if scheduler:
            self.wctl = nn.Sequential(nn.Linear(9, 16), nn.Tanh(), nn.Linear(16, 1))
            nn.init.zeros_(self.wctl[2].weight)
            nn.init.zeros_(self.wctl[2].bias)

    def w_values(self, x, meta):
        inp = torch.cat([x, meta], -1)
        return 2 * self.wctl(inp).sigmoid().squeeze(-1)

    def forward(self, x, meta):
        B = self.eff_B()
        xe = x * self.w_values(x, meta)[:, :, None] if self.scheduler else x
        xw = self.convolved(xe)
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


def build_s(seed, sw=.9, learnable=False, kernel_name='single', scheduler=False):
    torch.manual_seed(seed)
    return FlowS(sw, learnable, kernel_name, scheduler)


def train_hybrid(m, tr, val, seed, epochs, lr=E.LR):
    """pass1：截断通道给 core；pass2：完整通道给 B_raw + wctl。"""
    named = list(m.named_parameters())
    full_names = {n for n, _ in named if n == 'B_raw' or n.startswith('wctl.')}
    full_params = [p for n, p in named if n in full_names]
    core = [p for n, p in named if n not in full_names]
    opt_core = torch.optim.Adam(core, lr=lr)
    opt_full = torch.optim.Adam(full_params, lr=lr)
    gen = torch.Generator().manual_seed(seed + 5000)
    curve = []
    for epoch in range(epochs):
        m.train()
        m.detach_period = 5
        order = torch.randperm(512, generator=gen)
        for ch in order.split(E.BATCH):
            xb, mb, yb = tr[0][ch], tr[1][ch], tr[2][ch]
            opt_core.zero_grad(set_to_none=True)
            opt_full.zero_grad(set_to_none=True)
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
            nn.utils.clip_grad_norm_(full_params, 1.)
            opt_core.step()
            opt_full.step()
        if (epoch + 1) % 25 == 0 or epoch == 0:
            curve.append({'epoch': epoch + 1, **E.evaluate(m, val)})
    return curve


@torch.no_grad()
def w_profile(m, seed, task, T, n=1024):
    x, meta, _ = E.data_chain(seed * 10000 + 10, n, task, T)
    w = m.w_values(x, meta)
    tauA, tauC1, tauD, tauB, tauC2 = [max(1, min(T - 2, int(round(f * T)))) for f in (.05, .21, .39, .58, .76)]
    xn = x.abs().sum(-1) > 0
    return {'A': float(w[:, tauA].mean()), 'c1': float(w[:, tauC1].mean()),
            'distractor': float(w[:, tauD].mean()), 'B': float(w[:, tauB].mean()),
            'c2': float(w[:, tauC2].mean()), 'all': float(w.mean()),
            'active': float(w[xn].mean()) if bool(xn.any()) else float('nan')}


def run_chain_sched(kernel, task, seed, out, T=80, epochs=500):
    name = f'sched_{kernel}_{task}_T{T}_{seed}'
    p = out / (name + '.json')
    if p.exists():
        return
    m = build_s(seed, .9, learnable=True, kernel_name=kernel, scheduler=True)
    with torch.no_grad():
        m.B_raw.copy_(E.make_fixed_B_raw(seed))
    tr = E.data_chain(seed * 10000, 512, task, T)
    val = E.data_chain(seed * 10000 + 1, 256, task, T)
    start = time.perf_counter()
    curve = train_hybrid(m, tr, val, seed, epochs)
    te = E.data_chain(seed * 10000 + 10, 1024, task, T)
    final = E.evaluate(m, te)
    best = max(c['acc'] for c in curve) if curve else float('nan')
    row = {'name': name, 'phase': 'sched', 'kernel': kernel, 'task': task, 'T': T, 'seed': seed,
           'final': final['acc'], 'y0': final['y0'], 'y1': final['y1'], 'best_val': best,
           'w_profile': w_profile(m, seed, task, T), 'curve': curve, 'epochs': epochs,
           'seconds': time.perf_counter() - start, 'lr': E.LR}
    p.write_text(json.dumps(row, indent=2))
    print(f'{name}: acc={final["acc"]:.3f} best={best:.3f} w=' +
          ','.join(f'{k}:{v:.2f}' for k, v in row['w_profile'].items()) +
          f' {row["seconds"]:.0f}s', flush=True)


def _job(args):
    run_chain_sched(**args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kernels', nargs='+', default=['single', 'burst5', 'decay-slow'])
    ap.add_argument('--tasks', nargs='+', default=['xor', 'xorsw'])
    ap.add_argument('--Ts', nargs='+', type=int, default=[80])
    ap.add_argument('--seeds', nargs='+', type=int, default=E.SEEDS)
    ap.add_argument('--epochs', type=int, default=500)
    ap.add_argument('--out', default='results/sched_chain')
    ap.add_argument('--jobs', type=int, default=1)
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    jobs = [dict(kernel=k, task=t, seed=s, out=out, T=T, epochs=args.epochs)
            for k in args.kernels for t in args.tasks for T in args.Ts for s in args.seeds]
    print('sched jobs', len(jobs), 'out', out, flush=True)
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

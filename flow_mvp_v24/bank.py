"""V24C：Adaptive Temporal Compression（kernel bank selector）。

- α_t = softmax(MLP([x_t, meta_t]))（9→16→5）；a_t = α_t·K，ã_t = a_t/‖a_t‖（Σa²=1）；
- 写入 z_{t+τ} += ã_{t,τ}·B x_t；无 w_t；core K=5 不变；
- scheduler 与 B 走 hybrid full 通道，core 截断。
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

import experiment as E

ROOT = Path(__file__).parent
torch.set_num_threads(int(os.environ.get('FLOW_THREADS', '1')))
torch.use_deterministic_algorithms(True)

N, M, WRITE = E.N, E.M, E.WRITE


def _pad(a, L=5):
    a = list(a) + [0.0] * (L - len(a))
    return torch.tensor(a, dtype=torch.float32)


def _decay(r):
    a = torch.tensor([r ** k for k in range(5)])
    return _pad((a / a.norm()).tolist())


BANK = {
    'single': _pad([1.0]),
    'burst3': _pad([1 / math.sqrt(3)] * 3),
    'burst5': _pad([1 / math.sqrt(5)] * 5),
    'decay-fast': _decay(.5),
    'decay-slow': _decay(.85),
}
KMAT = torch.stack([BANK[k] for k in ('single', 'burst3', 'burst5', 'decay-fast', 'decay-slow')])


class FlowB(E.Flow):
    """E.Flow + kernel bank selector（不使用 E.Flow 的固定 kernel）。"""

    def __init__(self, sw=.9, learnable=False, bank=True):
        super().__init__(sw, learnable, kernel_name='single')
        self.bank = bank
        self.register_buffer('kmat', KMAT.clone())
        self.wbank = nn.Sequential(nn.Linear(9, 16), nn.Tanh(), nn.Linear(16, 5))
        nn.init.zeros_(self.wbank[2].weight)
        with torch.no_grad():
            self.wbank[2].bias.copy_(torch.tensor([4., 0., 0., 0., 0.]))

    def alpha(self, x, meta):
        return self.wbank(torch.cat([x, meta], -1)).softmax(-1)

    def mixed_kernel(self, x, meta):
        a = self.alpha(x, meta) @ self.kmat
        return a / (a.norm(dim=-1, keepdim=True) + 1e-12)

    def writes(self, x, meta):
        """无 in-place 的实现（in-place 切片赋值会让 autograd 每次清零整张量，慢 ~8×）。"""
        B = self.eff_B()
        a = self.mixed_kernel(x, meta)
        T = x.shape[1]
        w0 = (x @ B).reshape(len(x), T, 4, self.m)
        outs = [a[:, :, 0][:, :, None, None] * w0]
        for tau in range(1, 5):
            if tau < T:
                shifted = a[:, :T - tau, tau][:, :, None, None] * w0[:, :T - tau]
                outs.append(F.pad(shifted, (0, 0, 0, 0, tau, 0)))
        return sum(outs)

    def forward(self, x, meta):
        wr = self.writes(x, meta)
        h = x.new_zeros(len(x), 4, self.m)
        w = (self.W * self.mask).reshape(4, self.m, 4, self.m)
        for t in range(x.shape[1]):
            if self.training and self.detach_window and t and t % self.detach_period == 0:
                h = h.detach()
            ctl = meta[:, t]
            a = .2 * self.hold(ctl).sigmoid()
            gate = self.route(torch.cat([ctl, h.sum(-1) / self.m], 1)).sigmoid().reshape(-1, 4, 4)
            rec = (torch.einsum('bsi,sidj->bsdj', h, w) * gate[:, :, :, None]).sum(1)
            cand = torch.tanh(rec + wr[:, t])
            h = (1 - a[:, :, None]) * h + a[:, :, None] * cand
        return torch.cat([F.linear(h[:, 2], self.headX.weight, self.headX.bias),
                          F.linear(h[:, 3], self.headY.weight, self.headY.bias)], 1)


def build_bank(seed, sw=.9, learnable=False):
    torch.manual_seed(seed)
    return FlowB(sw, learnable)


def train_hybrid(m, tr, val, seed, epochs, lr=E.LR):
    named = list(m.named_parameters())
    full_names = {n for n, _ in named if n == 'B_raw' or n.startswith('wbank.')}
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


def _profile(m, x, meta, taus):
    with torch.no_grad():
        al = m.alpha(x, meta)
        ak = m.mixed_kernel(x, meta)
        idx = torch.arange(5, dtype=torch.float32)
    out = {}
    for name, t in taus.items():
        out[name] = {'alpha': [float(v) for v in al[:, t].mean(0)],
                     'L_eff': float((ak[:, t] ** 2 * idx).sum(-1).mean())}
    return out


def chain_profile(m, seed, task, T, n=1024):
    x, meta, _ = E.data_chain(seed * 10000 + 10, n, task, T)
    tauA, tauC1, tauD, tauB, tauC2 = [max(1, min(T - 2, int(round(f * T)))) for f in (.05, .21, .39, .58, .76)]
    return _profile(m, x, meta, {'A': tauA, 'c1': tauC1, 'distractor': tauD, 'B': tauB, 'c2': tauC2})


def delay_profile(m, seed, task, T, n=1024):
    x, meta, _ = E.data(seed * 10000 + task * 100 + 10, n, task, T)
    return _profile(m, x, meta, {'A': 0, 'B': 1})


@torch.no_grad()
def _states_b(m, x, meta, times):
    wr = m.writes(x, meta)
    w = (m.W * m.mask).reshape(4, M, 4, M)
    h = x.new_zeros(len(x), 4, M)
    out = {}
    for t in range(x.shape[1]):
        ctl = meta[:, t]
        a = .2 * m.hold(ctl).sigmoid()
        gate = m.route(torch.cat([ctl, h.sum(-1) / M], 1)).sigmoid().reshape(-1, 4, 4)
        rec = (torch.einsum('bsi,sidj->bsdj', h, w) * gate[:, :, :, None]).sum(1)
        cand = torch.tanh(rec + wr[:, t])
        h = (1 - a[:, :, None]) * h + a[:, :, None] * cand
        if t in times:
            out[t] = h.clone()
    return out


@torch.no_grad()
def chain_probe(m, seed, task, T, n=1024, times=None):
    x, meta, y = E.data_chain(seed * 10000 + 10, n, task, T)
    times = times or sorted({T // 2, T - 1})
    h = _states_b(m, x, meta, set(times))
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


@torch.no_grad()
def delay_probe(m, seed, task, T, n=1024, times=None):
    x, meta, y = E.data(seed * 10000 + task * 100 + 10, n, task, T)
    times = times or sorted({5, T // 2, T - 1})
    h = _states_b(m, x, meta, set(times))
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


def run_chain(seed, task, out, T=80, epochs=500):
    name = f'bank_{task}_T{T}_{seed}'
    p = out / (name + '.json')
    if p.exists():
        return
    m = build_bank(seed, .9, learnable=True)
    with torch.no_grad():
        m.B_raw.copy_(E.make_fixed_B_raw(seed))
    tr = E.data_chain(seed * 10000, 512, task, T)
    val = E.data_chain(seed * 10000 + 1, 256, task, T)
    start = time.perf_counter()
    curve = train_hybrid(m, tr, val, seed, epochs)
    te = E.data_chain(seed * 10000 + 10, 1024, task, T)
    final = E.evaluate(m, te)
    best = max(c['acc'] for c in curve) if curve else float('nan')
    row = {'name': name, 'phase': 'bank', 'task': task, 'T': T, 'seed': seed,
           'final': final['acc'], 'y0': final['y0'], 'y1': final['y1'], 'best_val': best,
           'profile': chain_profile(m, seed, task, T), 'probe': chain_probe(m, seed, task, T),
           'curve': curve, 'epochs': epochs, 'seconds': time.perf_counter() - start, 'lr': E.LR}
    p.write_text(json.dumps(row, indent=2))
    pr = row['profile']
    print(f'{name}: acc={final["acc"]:.3f} best={best:.3f} '
          f'L=[A{pr["A"]["L_eff"]:.2f} c1{pr["c1"]["L_eff"]:.2f} d{pr["distractor"]["L_eff"]:.2f} '
          f'B{pr["B"]["L_eff"]:.2f} c2{pr["c2"]["L_eff"]:.2f}] {row["seconds"]:.0f}s', flush=True)


def run_delay(seed, task, T, out, epochs=65):
    name = f'bank_delay_T{T}_t{task}_{seed}'
    p = out / (name + '.json')
    if p.exists():
        return
    m = build_bank(seed, .9, learnable=False)
    E.make_fixed_B(m, seed, 0.0)
    tr = E.data(seed * 10000 + task * 100, 512, task, T)
    val = E.data(seed * 10000 + task * 100 + 1, 256, task, T)
    start = time.perf_counter()
    curve = train_hybrid(m, tr, val, seed, epochs)
    final = E.evaluate(m, E.data(seed * 10000 + task * 100 + 10, 1024, task, T))
    row = {'name': name, 'phase': 'bank_delay', 'task': task, 'T': T, 'seed': seed,
           'final': final['acc'], 'y0': final['y0'], 'y1': final['y1'], 'curve': curve,
           'profile': delay_profile(m, seed, task, T), 'probe': delay_probe(m, seed, task, T),
           'epochs': epochs,
           'seconds': time.perf_counter() - start, 'lr': E.LR}
    p.write_text(json.dumps(row, indent=2))
    print(f'{name}: acc={final["acc"]:.3f} L_A={row["profile"]["A"]["L_eff"]:.2f} '
          f'L_B={row["profile"]["B"]["L_eff"]:.2f} {row["seconds"]:.0f}s', flush=True)


def _job(j):
    if j.pop('kind') == 'chain':
        run_chain(**j)
    else:
        run_delay(**j)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['chain', 'delay'], default='chain')
    ap.add_argument('--tasks', nargs='+', default=None)
    ap.add_argument('--Ts', nargs='+', type=int, default=[80])
    ap.add_argument('--seeds', nargs='+', type=int, default=E.SEEDS)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--jobs', type=int, default=1)
    args = ap.parse_args()
    if args.phase == 'chain':
        out = ROOT / (args.out or 'results/bank_chain')
        tasks = args.tasks or ['xor', 'xorsw']
        epochs = args.epochs or 500
        jobs = [dict(kind='chain', seed=s, task=t, out=out, T=args.Ts[0], epochs=epochs)
                for t in tasks for s in args.seeds]
    else:
        out = ROOT / (args.out or 'results/bank_delay')
        tasks = [int(t) for t in (args.tasks or [0, 1, 2, 3])]
        epochs = args.epochs or 65
        jobs = [dict(kind='delay', seed=s, task=t, T=T, out=out, epochs=epochs)
                for T in args.Ts for t in tasks for s in args.seeds]
    out.mkdir(parents=True, exist_ok=True)
    print('bank', args.phase, 'jobs', len(jobs), 'out', out, flush=True)
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

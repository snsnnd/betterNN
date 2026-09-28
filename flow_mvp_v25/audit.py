"""第二十五轮（V24D-Phase 0）：Temporal Strategy 可发现性审计。

- A：bank selector 初始化位置对照（single/burst5/decay-slow × 65/500 epochs）；
- B：共享标量 λ（几何核 a_τ∝λ^τ，L2 归一）初始化对照；
- F：固定 kernel 500-epoch 对照；
- 诊断：α/λ 位移、初始 full-BPTT 梯度方向与方差、冻结参数的条件 loss landscape。
协议同 V24：延迟 task0、T=80、B fixed-disjoint、core K=5、scheduler hybrid full。
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / 'flow_mvp_v24'))
import experiment as E  # noqa: E402
import bank as B  # noqa: E402

torch.set_num_threads(int(os.environ.get('FLOW_THREADS', '1')))
torch.use_deterministic_algorithms(True)

N, M = E.N, E.M
LR, BATCH = E.LR, E.BATCH
KNAMES = ('single', 'burst3', 'burst5', 'decay-fast', 'decay-slow')
LOGIT_INIT = 4.0

ARMS = {
    'A1': dict(type='bank', init='single', epochs=65),
    'A2': dict(type='bank', init='burst5', epochs=65),
    'A3': dict(type='bank', init='decay-slow', epochs=65),
    'A4': dict(type='bank', init='single', epochs=500),
    'B1': dict(type='shared', lam0=0.05, epochs=65),
    'B2': dict(type='shared', lam0=0.6, epochs=65),
    'B3': dict(type='shared', lam0=0.05, epochs=500),
    'F1': dict(type='fixed', kernel='single', epochs=500),
    'F2': dict(type='fixed', kernel='burst5', epochs=500),
    'F3': dict(type='fixed', kernel='decay-slow', epochs=500),
}
FIXED_KERNEL = {
    'single': torch.tensor([1., 0., 0., 0., 0.]),
    'burst5': torch.full((5,), 1 / math.sqrt(5)),
    'decay-slow': (lambda a: a / a.norm())(torch.tensor([0.85 ** k for k in range(5)])),
}


def geom_kernel(lam):
    """a_τ = c λ^τ，L2 归一；λ→0 为 single，λ→1（有限窗）为 burst5。

    入参为 tensor 时保持计算图（scheduler 需要梯度）。
    """
    lam = torch.as_tensor(lam, dtype=torch.float32).clamp(0.0, 1 - 1e-7)
    a = lam ** torch.arange(5, dtype=torch.float32)
    return a / a.norm()


def interp_kernel(k0, k1, g):
    a = (1 - g) * k0 + g * k1
    return a / a.norm()


class BankAudit(B.FlowB):
    """V24C bank selector + 可指定初始化 + landscape 覆盖钩子。"""

    def __init__(self, sw=.9, learnable=False, init='single'):
        super().__init__(sw, learnable)
        self.init_name = init
        self._a_override = None
        with torch.no_grad():
            self.wbank[2].bias.zero_()
            self.wbank[2].bias[KNAMES.index(init)] = LOGIT_INIT

    def mixed_kernel(self, x, meta):
        if self._a_override is not None:
            a = self._a_override.to(x.dtype)
            return (a / a.norm()).expand(x.shape[0], x.shape[1], -1)
        return super().mixed_kernel(x, meta)


class SharedLambda(B.FlowB):
    """共享标量 λ：所有事件/时间同一个几何核（无条件化 MLP）。"""

    def __init__(self, sw=.9, learnable=False, lam0=0.05):
        super().__init__(sw, learnable)
        del self.wbank  # 不条件化：去掉 bank 分支（RNG 已按同序消耗，core 与 A 臂对齐）
        self.lam0 = lam0
        self.s = nn.Parameter(torch.tensor(math.log(lam0 / (1 - lam0)), dtype=torch.float32))
        self._a_override = None

    def lam(self):
        return torch.sigmoid(self.s)

    def kernel_a(self, lam=None):
        return geom_kernel(self.lam() if lam is None else lam)

    def mixed_kernel(self, x, meta):
        if self._a_override is not None:
            a = self._a_override
            return (a / a.norm()).expand(x.shape[0], x.shape[1], -1)
        return self.kernel_a().expand(x.shape[0], x.shape[1], -1)


def full_names_of(m):
    return {n for n, _ in m.named_parameters()
            if n == 'B_raw' or n == 's' or n.startswith('wbank.')}


@torch.no_grad()
def probe_profile(m, seed, task, T, n=256):
    x, meta, _ = E.data(seed * 10000 + task * 100 + 10, n, task, T)
    km = m.mixed_kernel(x, meta)
    al = None if isinstance(m, SharedLambda) else m.alpha(x, meta)
    out = {}
    for name, t in (('A', 0), ('B', 1)):
        a = km[:, t].mean(0)
        row = {'k': [round(float(v), 4) for v in a],
               'L_eff': float((a.pow(2) * torch.arange(5.)).sum())}
        if al is None:
            row['lambda'] = round(float(m.lam()), 6)
        else:
            row['alpha'] = [round(float(v), 4) for v in al[:, t].mean(0)]
        out[name] = row
    return out


def displacement(m, p0, p1):
    if isinstance(m, SharedLambda):
        return {'d_lambda': abs(p1['A']['lambda'] - p0['A']['lambda'])}
    l1 = [float(np.abs(np.array(p1[e]['alpha']) - np.array(p0[e]['alpha'])).sum())
          for e in ('A', 'B')]
    return {'d_alpha_L1_mean': float(np.mean(l1)), 'd_alpha_L1': l1}


def grad_stats(m, seed, task, T, nbatch=4):
    """初始点梯度：截断应为 0；full BPTT 记录方向与 batch 间方差。"""
    tr = E.data(seed * 10000 + task * 100, 512, task, T)
    named = list(m.named_parameters())
    fnames = full_names_of(m)
    fnamed = [(n, p) for n, p in named if n in fnames]
    fparams = [p for _, p in fnamed]
    pos = {n: i for i, (n, _) in enumerate(fnamed)}
    out = {'trunc': {}, 'full': {}}

    m.train()
    m.detach_window = True
    m.zero_grad(set_to_none=True)
    F.binary_cross_entropy_with_logits(m(tr[0][:128], tr[1][:128]), tr[2][:128]).backward()
    for n, p in named:
        if n in fnames:
            out['trunc'][n] = float(p.grad.norm()) if p.grad is not None else 0.0
    m.zero_grad(set_to_none=True)

    m.eval()
    m.detach_window = False
    gs = []
    for b in range(nbatch):
        sl = slice(b * 128, (b + 1) * 128)
        l = F.binary_cross_entropy_with_logits(m(tr[0][sl], tr[1][sl]), tr[2][sl])
        g = torch.autograd.grad(l, fparams, allow_unused=True)
        gs.append([None if x is None else x.detach().clone() for x in g])

    if isinstance(m, SharedLambda):
        s0 = float(m.s.detach())
        ds = float(torch.sigmoid(torch.tensor(s0)) * (1 - torch.sigmoid(torch.tensor(s0))))
        sg = torch.tensor([float(g[pos['s']]) for g in gs])
        lg = sg * ds
        out['full'] = {'s0': s0, 'ds_dsigma': ds,
                       's_mean': float(sg.mean()), 's_std': float(sg.std(unbiased=False)),
                       'lambda_mean': float(lg.mean()), 'lambda_std': float(lg.std(unbiased=False)),
                       'per_batch_lambda': [float(v) for v in lg],
                       'n_negative': int((lg < 0).sum())}
    else:
        bg = torch.stack([g[pos['wbank.2.bias']] for g in gs])
        out['full'] = {'bias_mean': [float(v) for v in bg.mean(0)],
                       'bias_std': [float(v) for v in bg.std(0, unbiased=False)],
                       'bias_lambda_mean': float(bg.mean(0)[4]),
                       'bias_burst5_mean': float(bg.mean(0)[2]),
                       'n_negative_burst5': int((bg[:, 2] < 0).sum()),
                       'n_negative_dslow': int((bg[:, 4] < 0).sum())}
    return out


@torch.no_grad()
def eval_loss_acc(m, d, batch=256):
    m.eval()
    tot, n, correct = 0.0, 0, 0
    for ids in torch.arange(len(d[0])).split(batch):
        out = m(d[0][ids], d[1][ids])
        tot += float(F.binary_cross_entropy_with_logits(out, d[2][ids], reduction='sum'))
        n += len(ids)
        correct += int(((out > 0) == d[2][ids].bool()).all(1).sum())
    return tot / n, correct / n


def scan(m, d, mode, npts=21):
    """冻结参数，只扫 kernel（条件 landscape）。mode: lambda/burst5/decay-slow。"""
    grid = np.linspace(0, 1, npts)
    rows = []
    with torch.no_grad():
        for p in grid:
            if mode == 'lambda':
                m._a_override = geom_kernel(float(p))
            elif mode == 'burst5':
                m._a_override = interp_kernel(FIXED_KERNEL['single'], FIXED_KERNEL['burst5'], float(p))
            else:
                m._a_override = interp_kernel(FIXED_KERNEL['single'], FIXED_KERNEL['decay-slow'], float(p))
            loss, acc = eval_loss_acc(m, d)
            rows.append({'x': float(p), 'loss': loss, 'acc': acc})
    m._a_override = None
    return rows


def scan_modes(m, val):
    if isinstance(m, SharedLambda):
        return {'lambda': scan(m, val, 'lambda')}
    return {'burst5': scan(m, val, 'burst5'), 'decay-slow': scan(m, val, 'decay-slow')}


def train_hybrid(m, tr, val, seed, epochs, task, T, every=25):
    """V24B/C 协议：core 截断 + scheduler full BPTT 双通道、分别 clip。"""
    named = list(m.named_parameters())
    fnames = full_names_of(m)
    full_params = [p for n, p in named if n in fnames]
    core = [p for n, p in named if n not in fnames]
    opt_core = torch.optim.Adam(core, lr=LR)
    opt_full = torch.optim.Adam(full_params, lr=LR)
    gen = torch.Generator().manual_seed(seed + 5000)
    curve = []
    for epoch in range(epochs):
        m.train()
        m.detach_period = 5
        order = torch.randperm(512, generator=gen)
        for ch in order.split(BATCH):
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
                if n in fnames:
                    p.grad = gf[gi]
                    gi += 1
                else:
                    p.grad = gt.get(n)
            nn.utils.clip_grad_norm_(core, 1.)
            nn.utils.clip_grad_norm_(full_params, 1.)
            opt_core.step()
            opt_full.step()
        if (epoch + 1) % every == 0 or epoch == 0:
            curve.append({'epoch': epoch + 1, **E.evaluate(m, val),
                          'prof': probe_profile(m, seed, task, T, n=128)})
    return curve


def run_arm(arm, seed, task, T, out, epochs=None):
    spec = ARMS[arm]
    name = f'audit_{arm}_T{T}_t{task}_{seed}'
    p = out / (name + '.json')
    if p.exists():
        return
    ep = int(epochs or spec['epochs'])
    torch.manual_seed(seed)
    start = time.perf_counter()
    if spec['type'] == 'fixed':
        m = E.build(seed, .9, learnable=False, kernel_name=spec['kernel'])
        E.make_fixed_B(m, seed, 0.0)
        curve = E.train_fixed(m, seed, task, T, ep)
        final = E.evaluate(m, E.data(seed * 10000 + task * 100 + 10, 1024, task, T))
        row = {'name': name, 'arm': arm, 'type': 'fixed', 'kernel': spec['kernel'],
               'T': T, 'task': task, 'seed': seed, 'epochs': ep,
               'final': final['acc'], 'y0': final['y0'], 'y1': final['y1'],
               'best_val': max(c['acc'] for c in curve), 'curve': curve,
               'seconds': time.perf_counter() - start}
    else:
        if spec['type'] == 'bank':
            m = BankAudit(.9, learnable=False, init=spec['init'])
            meta = {'init': spec['init']}
        else:
            m = SharedLambda(.9, learnable=False, lam0=spec['lam0'])
            meta = {'lam0': spec['lam0']}
        E.make_fixed_B(m, seed, 0.0)
        tr = E.data(seed * 10000 + task * 100, 512, task, T)
        val = E.data(seed * 10000 + task * 100 + 1, 256, task, T)
        prof0 = probe_profile(m, seed, task, T)
        gstats = grad_stats(m, seed, task, T)
        scan0 = scan_modes(m, val)
        curve = train_hybrid(m, tr, val, seed, ep, task, T)
        final = E.evaluate(m, E.data(seed * 10000 + task * 100 + 10, 1024, task, T))
        prof1 = probe_profile(m, seed, task, T)
        scan1 = scan_modes(m, val)
        row = {'name': name, 'arm': arm, 'type': spec['type'], **meta,
               'T': T, 'task': task, 'seed': seed, 'epochs': ep,
               'final': final['acc'], 'y0': final['y0'], 'y1': final['y1'],
               'best_val': max(c['acc'] for c in curve), 'curve': curve,
               'profile_init': prof0, 'profile_end': prof1,
               'displacement': displacement(m, prof0, prof1), 'grad_init': gstats,
               'scan_init': scan0, 'scan_end': scan1,
               'seconds': time.perf_counter() - start}
    p.write_text(json.dumps(row, indent=2))
    extra = (f" d={row.get('displacement')}" if 'displacement' in row else '')
    print(f"{name}: final={row['final']:.3f} best={row['best_val']:.3f}{extra} "
          f"{row['seconds']:.0f}s", flush=True)


def _job(j):
    run_arm(**j)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='+', default=list(ARMS))
    ap.add_argument('--tasks', nargs='+', type=int, default=[0])
    ap.add_argument('--T', type=int, default=80)
    ap.add_argument('--seeds', nargs='+', type=int, default=[11, 22, 33])
    ap.add_argument('--epochs', type=int, default=None, help='仅 smoke：覆盖所有 arm 的 epochs')
    ap.add_argument('--out', default='results/audit')
    ap.add_argument('--jobs', type=int, default=1)
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    jobs = [dict(arm=a, seed=s, task=t, T=args.T, out=out, epochs=args.epochs)
            for a in args.arms for t in args.tasks for s in args.seeds]
    print('audit jobs', len(jobs), 'out', out, flush=True)
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

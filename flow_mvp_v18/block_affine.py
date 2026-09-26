"""第十八轮：非线性 Flow 的 block-affine 可压缩性（诊断，无需训练）。

用第十五轮训练好的 Flow-v2 检查点，在测试分布上比较：
- Flow-v2 串行真值（参考）
- PreRoute（v17 可 scan 参考）
- 块仿射 + 迭代重线性化（zero / preroute 初始估计，K=0..3，L=1/2/4/5/10）
输出每个案例的准确率、边界状态误差 E_h，以及部分配置的块 Jacobian sigma_max。
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.func import jvp, vjp

ROOT = Path(__file__).parent
torch.set_num_threads(int(os.environ.get('FLOW_THREADS', '1')))
torch.use_deterministic_algorithms(True)
N = 256
SEEDS = [11, 22, 33, 44, 55]
TASKS = [0, 1, 2, 3]
LS = [1, 2, 4, 5, 10]
KS = [0, 1, 2, 3]
INITS = ['zero', 'preroute']
BATCH = 256
V15 = (ROOT / '../flow_mvp_v15/results').resolve()
V17 = (ROOT / '../flow_mvp_v17/results').resolve()
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


class PreRoute(nn.Module):
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
        hs = []
        for t in range(x.shape[1]):
            ctl = meta[:, t]
            a = .2 * self.hold(ctl).sigmoid()
            pad = x.new_zeros(len(x), 2)
            q = torch.cat([ctl, x[:, t], pad], 1)
            gate = self.route(q).sigmoid().reshape(-1, 4, 4)
            rec = (torch.einsum('bsi,sidj->bsdj', h, w) * gate[:, :, :, None]).sum(1)
            h = (1 - a[:, :, None]) * h + a[:, :, None] * (rec + (x[:, t] @ self.B).reshape(-1, 4, self.m))
            if record:
                hs.append(h)
        if record:
            return h, torch.stack(hs, 1)
        return h


def load_flow(seed):
    m = FlowV2().to(DEVICE)
    ck = torch.load(V15 / f'o0_flowv2_r12.5_{seed}_stage3.pt', map_location='cpu', weights_only=False)
    m.load_state_dict(ck['state'])
    m.eval()
    return m


def load_preroute(seed):
    m = PreRoute().to(DEVICE)
    ck = torch.load(V17 / f'o0_preroute_r12.5_{seed}_stage3.pt', map_location='cpu', weights_only=False)
    m.load_state_dict(ck['state'])
    m.eval()
    return m


def readout(m, h):
    return torch.cat([F.linear(h[:, 2], m.headX.weight, m.headX.bias),
                      F.linear(h[:, 3], m.headY.weight, m.headY.bias)], 1)


def block_map(m, h, xb, mb):
    w = (m.W * m.mask).reshape(4, m.m, 4, m.m)
    for t in range(xb.shape[1]):
        ctl = mb[:, t]
        a = .2 * m.hold(ctl).sigmoid()
        gate = m.route(torch.cat([ctl, h.sum(-1) / m.m], 1)).sigmoid().reshape(-1, 4, 4)
        rec = (torch.einsum('bsi,sidj->bsdj', h, w) * gate[:, :, :, None]).sum(1)
        cand = torch.tanh(rec + (xb[:, t] @ m.B).reshape(-1, 4, m.m))
        h = (1 - a[:, :, None]) * h + a[:, :, None] * cand
    return h


def boundaries(m, x, meta, L):
    steps = x.shape[1]
    B = steps // L
    h = x.new_zeros(len(x), 4, m.m)
    hs = [h]
    for i in range(B):
        h = block_map(m, h, x[:, i * L:(i + 1) * L], meta[:, i * L:(i + 1) * L])
        hs.append(h)
    return hs


def preroute_boundaries(pm, x, meta, L):
    steps = x.shape[1]
    B = steps // L
    with torch.no_grad():
        _, traj = pm(x, meta, record=True)
    ids = [min(i * L - 1, steps - 1) for i in range(1, B + 1)]
    hs = [x.new_zeros(len(x), 4, pm.m)]
    for i in ids:
        hs.append(traj[:, i])
    return hs


def eh(est, true):
    vals = []
    for a, b in zip(est[1:], true[1:]):
        vals.append(float((a - b).norm(dim=(1, 2)).mean() / (b.norm(dim=(1, 2)).mean() + 1e-9)))
    return float(np.mean(vals))


def compose(m, x, meta, L, p):
    steps = x.shape[1]
    B = steps // L
    h = x.new_zeros(len(x), 4, m.m)
    states = [h]
    for i in range(B):
        xb = x[:, i * L:(i + 1) * L]
        mb = meta[:, i * L:(i + 1) * L]
        pi = p[i]
        f = lambda hh: block_map(m, hh, xb, mb)
        ci = f(pi)
        if i == 0:
            h = ci
        else:
            _, jv = jvp(f, (pi,), (h - pi,))
            h = ci + jv
        states.append(h)
    return states


def acc_from_state(m, h, y):
    return float(((readout(m, h) > 0) == y.bool()).all(1).float().mean())


def sigma_max(m, h, xb, mb, iters=8, gen=None):
    f = lambda hh: block_map(m, hh, xb, mb)
    v = torch.randn(h.shape, generator=gen or torch.Generator().manual_seed(0))
    v = v / (v.norm() + 1e-12)
    val = 0.0
    for _ in range(iters):
        _, jv = jvp(f, (h,), (v,))
        _, vjp_fn = vjp(f, h)
        vj = vjp_fn(jv)[0]
        val = float(jv.norm())
        v = vj / (vj.norm() + 1e-12)
    return val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--tasks', nargs='+', type=int, default=TASKS)
    ap.add_argument('--steps', nargs='+', type=int, default=[20, 40])
    ap.add_argument('--ls', nargs='+', type=int, default=LS)
    ap.add_argument('--ks', nargs='+', type=int, default=KS)
    ap.add_argument('--inits', nargs='+', default=INITS)
    ap.add_argument('--batch', type=int, default=BATCH)
    ap.add_argument('--out', default='block_affine.json')
    ap.add_argument('--sigma-diagnostic', action='store_true')
    args = ap.parse_args()
    rows = []
    t0 = time.perf_counter()
    for seed in args.seeds:
        fm = load_flow(seed)
        pm = load_preroute(seed)
        for task in args.tasks:
            for steps in args.steps:
                x_all, meta_all, y_all = data(seed * 10000 + task * 100 + 10, 1024, task, steps)
                x, meta, y = x_all[:args.batch], meta_all[:args.batch], y_all[:args.batch]
                for L in args.ls:
                    if steps % L:
                        continue
                    with torch.no_grad():
                        true = boundaries(fm, x, meta, L)
                        flow_acc = acc_from_state(fm, true[-1], y)
                        pre_acc = acc_from_state(pm, pm(x, meta), y)
                        pre_states = preroute_boundaries(pm, x, meta, L) if 'preroute' in args.inits else None
                    rows.append({'seed': seed, 'task': task, 'steps': steps, 'L': L, 'init': 'flow',
                                 'K': None, 'acc': flow_acc, 'Eh': 0.0})
                    rows.append({'seed': seed, 'task': task, 'steps': steps, 'L': L, 'init': 'preroute_ref',
                                 'K': None, 'acc': pre_acc, 'Eh': 0.0})
                    for init in args.inits:
                        p = [x.new_zeros(len(x), 4, fm.m)] * (steps // L + 1)
                        if init == 'preroute':
                            p = pre_states
                        if init == 'oracle':
                            p = [t.detach() for t in true]
                        for K in args.ks:
                            est = compose(fm, x, meta, L, p)
                            acc = acc_from_state(fm, est[-1], y)
                            rows.append({'seed': seed, 'task': task, 'steps': steps, 'L': L, 'init': init,
                                         'K': K, 'acc': acc, 'Eh': eh(est, [t.detach() for t in true])})
                            p = est
                    print(f"seed {seed} task {task} steps {steps} L {L} done", flush=True)
                    if args.sigma_diagnostic and steps == 20 and L in (4,) and ('zero' in args.inits):
                        gen = torch.Generator().manual_seed(seed * 100 + task)
                        sm = []
                        for i in range(steps // L):
                            sm.append(sigma_max(fm, true[i], x[:, i * L:(i + 1) * L], meta[:, i * L:(i + 1) * L], gen=gen))
                        rows.append({'seed': seed, 'task': task, 'steps': steps, 'L': L, 'init': 'sigma_max',
                                     'K': None, 'acc': None, 'Eh': 0.0, 'sigma_max_mean': float(np.mean(sm))})
    (ROOT / args.out).write_text(json.dumps({'rows': rows, 'seconds': time.perf_counter() - t0}, indent=2))
    print('wrote', args.out, len(rows), 'rows in', round(time.perf_counter() - t0, 1), 's')


if __name__ == '__main__':
    main()

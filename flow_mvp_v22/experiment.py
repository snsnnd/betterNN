"""第二十二轮：Long-range Credit Attribution。

- 逐组信用窗口：每个 step 用同一参数快照做两条通道——
  截断通道（period=5）给 T 组、完整通道（20 步）给 Full 组；
- arms = {W, Route, Hold} 的 2³ 子集拿 Full；B 恒可训练恒 Full；
- 除信用窗口外与 V21 hybrid 完全一致（双 optimizer、分别 clip、RNG 调用顺序）；
- 训练中每 stage 记录逐组 credit_stats（period 5/10/20 的 cos/ratio）。
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
V21 = (ROOT / '../flow_mvp_v21').resolve()
torch.set_num_threads(int(os.environ.get('FLOW_THREADS', '1')))
torch.use_deterministic_algorithms(True)

N = 256
M = N // 4
WRITE = 2 * M                             # role0 ∪ role1
BETA = .7 * math.sqrt(M)                  # 5.6
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 110]
EPOCHS = 65
LR = .003
BATCH = 128
THRESHOLD = .9
BUFFER_PER_TASK = 32
ORDERS = {'o0': [0, 1, 2, 3], 'o1': [3, 2, 1, 0], 'o2': [1, 3, 0, 2]}
RATIOS = {0: '0', .125: '12.5'}
# arm -> core groups getting full 20-step credit（B/readout 恒 Full）
CREDIT = {
    'B': (),
    'B+W': ('W',),
    'B+R': ('route',),
    'B+H': ('hold',),
    'B+W+R': ('W', 'route'),
    'B+W+H': ('W', 'hold'),
    'B+R+H': ('route', 'hold'),
    'All': ('W', 'route', 'hold'),
}
ARMS = list(CREDIT)
GROUPS = ('W', 'route', 'hold', 'readout', 'B')
ALWAYS_FULL = ('B', 'readout')


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


def full_set_of(arm):
    return frozenset(ALWAYS_FULL) | frozenset(CREDIT[arm])


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


class Flow(nn.Module):
    """Flow-v2 + 可训练 B（仅 role0∪role1）。与 V21 逐位同构。"""

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
    """V22 所有 arm 都是 learnable B + overlap 初始化（与 V21 learnable 相同）。"""
    m = build(seed, sw, learnable=True)
    raw = _overlap_B(seed, 1.0)[:, :WRITE].clone()
    with torch.no_grad():
        m.B_raw.copy_(raw)
    return m


def b_metrics(B):
    b = B[:, :WRITE].detach()
    a, c = b[0], b[1]
    na, nc = a.norm() + 1e-12, c.norm() + 1e-12
    o_l1 = float((a * c).abs().sum() / (na * nc))
    cos = float((a @ c) / (na * nc))
    ta = set(torch.topk(a.abs(), 16).indices.tolist())
    tc = set(torch.topk(c.abs(), 16).indices.tolist())
    jac = len(ta & tc) / max(len(ta | tc), 1)
    return {'O_B': o_l1, 'cos_B': cos, 'jaccard16': jac}


def _flat_group_grads(m):
    out = {}
    for g in GROUPS:
        vec = [p.grad.reshape(-1) for n, p in m.named_parameters()
               if group_of(n) == g and p.grad is not None]
        out[g] = torch.cat(vec) if vec else None
    return out


def credit_stats(m, x, meta, y, periods=(5, 10, 20)):
    """逐组：period p 的梯度相对 20 步梯度的 cos 与模长比。period=20 即完整图。"""
    was_train = m.training
    m.train()
    grads = {}
    for p in periods:
        m.zero_grad(set_to_none=True)
        if p >= x.shape[1]:
            m.detach_window = False
        else:
            m.detach_window = True
            m.detach_period = p
        F.binary_cross_entropy_with_logits(m(x, meta), y).backward()
        grads[p] = _flat_group_grads(m)
    m.detach_window = True
    m.detach_period = 5
    m.zero_grad(set_to_none=True)
    if not was_train:
        m.eval()
    gf = grads[max(periods)]
    out = {}
    for p in periods:
        for g in GROUPS:
            a, b = grads[p][g], gf[g]
            if a is None or b is None or float(b.norm()) < 1e-30:
                out[f'cos{p}_{g}'] = float('nan')
                out[f'ratio{p}_{g}'] = 0.0 if (a is None or float(a.norm()) < 1e-30) else float('nan')
            else:
                out[f'cos{p}_{g}'] = float(F.cosine_similarity(a, b, dim=0))
                out[f'ratio{p}_{g}'] = float(a.norm() / b.norm())
    return out


def train_stage(m, seed, task, epochs, lr, full_set, replay_n=0, buffer=None,
                trace_every=5, audit=True):
    tr = data(seed * 10000 + task * 100, 512, task)
    val = data(seed * 10000 + task * 100 + 1, 256, task)
    core = [p for n, p in m.named_parameters() if n != 'B_raw']
    opt_core = torch.optim.Adam(core, lr=lr)
    opt_b = torch.optim.Adam([m.B_raw], lr=lr)
    gen = torch.Generator().manual_seed(seed + task * 100 + 5000)
    gen_rep = torch.Generator().manual_seed(seed + task * 100 + 9000)
    cur_n = BATCH - replay_n
    has_replay = replay_n > 0 and buffer is not None and len(buffer[0]) > 0
    named = list(m.named_parameters())
    hh, trace = [], []
    for epoch in range(epochs):
        m.train()
        loss_sum = 0
        order = torch.randperm(512, generator=gen)
        for chunk in order[:4 * cur_n].split(cur_n):
            if has_replay:
                bx, bm, by = (torch.cat(b) for b in buffer)
                rid = torch.randint(len(bx), (replay_n,), generator=gen_rep)
                xb = torch.cat([tr[0][chunk], bx[rid]])
                mb = torch.cat([tr[1][chunk], bm[rid]])
                yb = torch.cat([tr[2][chunk], by[rid]])
            else:
                xb, mb, yb = tr[0][chunk], tr[1][chunk], tr[2][chunk]
            # 同一快照：截断通道（全参数） + 完整通道（Full 组）
            opt_core.zero_grad(set_to_none=True)
            opt_b.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(m(xb, mb), yb)
            loss.backward()
            g_trunc = {n: p.grad.detach().clone() for n, p in named if p.grad is not None}
            for p in m.parameters():
                p.grad = None
            m.eval()
            loss_full = F.binary_cross_entropy_with_logits(m(xb, mb), yb)
            fparams = [p for n, p in named if group_of(n) in full_set]
            fgrads = torch.autograd.grad(loss_full, fparams)
            m.train()
            gi = 0
            for n, p in named:
                if group_of(n) in full_set:
                    p.grad = fgrads[gi]
                    gi += 1
                else:
                    p.grad = g_trunc[n]
            nn.utils.clip_grad_norm_(core, 1.)
            nn.utils.clip_grad_norm_([m.B_raw], 1.)
            opt_core.step()
            opt_b.step()
            loss_sum += float(loss.detach()) / len(order[:4 * cur_n].split(cur_n))
        hh.append({'loss': loss_sum, 'validation': evaluate(m, val)})
        if (epoch + 1) % trace_every == 0 or epoch == 0:
            trace.append({'epoch': epoch + 1, **b_metrics(m.eff_B())})
    audit_stats = None
    if audit:
        xa, ma, ya = data(seed * 10000 + task * 100 + 10, 128, task)
        audit_stats = credit_stats(m, xa, ma, ya)
    return hh, trace, audit_stats


@torch.no_grad()
def evaluate(m, d):
    m.eval()
    correct = 0
    for ids in torch.arange(len(d[0])).split(128):
        x, meta, y = d[0][ids], d[1][ids], d[2][ids]
        correct += int(((m(x, meta) > 0) == y.bool()).all(1).sum())
    return correct / len(d[0])


def scores(m, seed):
    return [evaluate(m, data(seed * 10000 + t * 100 + 10, 1024, t)) for t in range(4)]


def metrics(matrix, order):
    n = len(order)
    acq = float(np.mean([matrix[i][order[i]] for i in range(n)]))
    final = float(np.mean(matrix[-1]))
    forget = float(np.mean([max(matrix[s][t] for s in range(i, n)) - matrix[-1][t]
                            for i, t in enumerate(order[:-1])]))
    bwt = float(np.mean([matrix[-1][t] - matrix[i][t] for i, t in enumerate(order[:-1])]))
    return acq, final, forget, bwt


def run_cl(arm, order_key, ratio, seed, out, epochs, sw=.9, save_ckpt=True):
    label = RATIOS[ratio]
    name = f'{arm}_{order_key}_r{label}_{seed}'
    p = out / (name + '.json')
    if p.exists():
        return
    order = ORDERS[order_key]
    replay_n = int(round(BATCH * ratio))
    full_set = full_set_of(arm)
    m = make_model(seed, sw)
    B0 = m.eff_B().clone()
    b0 = b_metrics(B0)
    buffer = ([], [], [])
    matrix, history, epochs_to_90, traces = [], {}, {}, {}
    stage_metrics = []
    start = time.perf_counter()
    for i, task in enumerate(order):
        rn = replay_n if i > 0 else 0
        hh, tr, audit_stats = train_stage(m, seed, task, epochs, LR, full_set, rn, buffer)
        history[str(task)] = hh
        traces[str(task)] = tr
        epochs_to_90[str(task)] = next((j + 1 for j, h in enumerate(hh) if h['validation'] >= THRESHOLD), None)
        matrix.append(scores(m, seed))
        Bcur = m.eff_B().clone()
        sm = {**b_metrics(Bcur),
              'D_B': float((Bcur - B0).norm() / (B0.norm() + 1e-12)),
              'B_raw_norm': float(m.B_raw.norm()),
              'credit': audit_stats}
        stage_metrics.append(sm)
        if save_ckpt:
            torch.save({'state': m.state_dict(), 'order': order_key, 'ratio': ratio, 'seed': seed,
                        'stage': i, 'task': task, 'arm': arm, 'full_set': sorted(full_set), 'sw': sw},
                       out / f'{name}_stage{i}.pt')
        if ratio > 0:
            trd = data(seed * 10000 + task * 100, 512, task)
            for k in range(3):
                buffer[k].append(trd[k][:BUFFER_PER_TASK])
        print(name, 'stage', i, [round(v, 3) for v in matrix[-1]], 'O_B', round(sm['O_B'], 4),
              flush=True)
    acq, final, forget, bwt = metrics(matrix, order)
    row = {'name': name, 'arm': arm, 'full_set': sorted(full_set), 'order': order_key, 'order_seq': order,
           'ratio': ratio, 'replay_n': replay_n, 'seed': seed, 'B0': b0, 'B_final': b_metrics(m.eff_B()),
           'b_trace': traces, 'stage_metrics': stage_metrics, 'matrix': matrix, 'history': history,
           'epochs_to_90': epochs_to_90, 'acquisition': acq, 'final_mean': final, 'forgetting': forget,
           'bwt': bwt, 'params': sum(p.numel() for p in m.parameters() if p.requires_grad),
           'seconds': time.perf_counter() - start, 'epochs': epochs, 'lr': LR}
    p.write_text(json.dumps(row, indent=2))


def run_single(arm, task, seed, out, epochs, sw=.9):
    name = f'single_{arm}_t{task}_{seed}'
    p = out / (name + '.json')
    if p.exists():
        return
    m = make_model(seed, sw)
    start = time.perf_counter()
    hh, tr, audit_stats = train_stage(m, seed, task, epochs, LR, full_set_of(arm))
    final = evaluate(m, data(seed * 10000 + task * 100 + 10, 1024, task))
    torch.save({'state': m.state_dict(), 'seed': seed, 'task': task, 'arm': arm, 'sw': sw},
               out / f'{name}.pt')
    row = {'name': name, 'arm': arm, 'full_set': sorted(full_set_of(arm)), 'task': task, 'seed': seed,
           'final': final, 'B_final': b_metrics(m.eff_B()), 'b_trace': tr, 'credit': audit_stats,
           'history': hh, 'seconds': time.perf_counter() - start, 'epochs': epochs, 'lr': LR}
    p.write_text(json.dumps(row, indent=2))


def _job(args):
    kind, kw = args
    if kind == 'cl':
        run_cl(**kw)
    else:
        run_single(**kw)


def credit_audit(seeds, out_dir):
    """Phase 0：在 V21 检查点上做逐组信用窗口审计（不训练）。"""
    rows = []
    for seed in seeds:
        m = make_model(seed)
        x, meta, y = data(seed * 10000 + 10, 256, 0)
        rows.append({'source': 'init', 'seed': seed, 'task': 0, **credit_stats(m, x, meta, y)})
        for task in range(4):
            p = V21 / 'results/A' / f'single_learnable_t{task}_{seed}.pt'
            if not p.exists():
                continue
            ck = torch.load(p, map_location='cpu')
            m = build(seed, learnable=True)
            m.load_state_dict(ck['state'])
            x, meta, y = data(seed * 10000 + task * 100 + 10, 256, task)
            rows.append({'source': 'single', 'seed': seed, 'task': task, **credit_stats(m, x, meta, y)})
        for ratio, label in RATIOS.items():
            for stage in range(4):
                p = V21 / 'results/A' / f'learnable_o0_r{label}_{seed}_stage{stage}.pt'
                if not p.exists():
                    continue
                ck = torch.load(p, map_location='cpu')
                m = build(seed, learnable=True)
                m.load_state_dict(ck['state'])
                task = int(ck['task'])
                x, meta, y = data(seed * 10000 + task * 100 + 10, 256, task)
                rows.append({'source': f'cl_r{label}', 'seed': seed, 'task': task, 'stage': stage,
                             **credit_stats(m, x, meta, y)})
        print('audit seed', seed, 'done', flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'credit_audit.json').write_text(json.dumps(rows, indent=2))
    for src in sorted({r['source'] for r in rows}):
        rs = [r for r in rows if r['source'] == src]
        print(src, {g: round(float(np.nanmean([r[f'cos5_{g}'] for r in rs])), 3) for g in GROUPS},
              'ratio5', {g: round(float(np.nanmean([r[f'ratio5_{g}'] for r in rs])), 3) for g in GROUPS})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='+', default=ARMS)
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--single-seeds', nargs='+', type=int, default=None)
    ap.add_argument('--orders', nargs='+', default=list(ORDERS))
    ap.add_argument('--ratios', nargs='+', type=float, default=list(RATIOS))
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--out', default='results/A')
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--no-single', action='store_true')
    ap.add_argument('--no-ckpt', action='store_true')
    ap.add_argument('--credit-audit', action='store_true')
    args = ap.parse_args()
    if args.credit_audit:
        credit_audit(args.seeds, ROOT / args.out)
        return
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    ratios = [r for r in RATIOS if r in args.ratios]
    print('arms', args.arms, 'ratios', ratios, 'epochs', args.epochs, flush=True)
    jobs = []
    for arm in args.arms:
        for seed in args.seeds:
            for order_key in args.orders:
                for ratio in ratios:
                    jobs.append(('cl', dict(arm=arm, order_key=order_key, ratio=ratio, seed=seed,
                                            out=out, epochs=args.epochs, save_ckpt=not args.no_ckpt)))
            if not args.no_single and seed in (args.single_seeds or args.seeds[:5]):
                for task in range(4):
                    jobs.append(('single', dict(arm=arm, task=task, seed=seed, out=out, epochs=args.epochs)))
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

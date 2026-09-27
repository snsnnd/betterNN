"""第二十轮：B 输入拓扑 × 底网动力学。Phase A（主族）/ B（shared-pool overlap）/ C（s_W）。

不改 Flow-v2 架构，只改 B 的支撑/符号/归一化与 W 初始尺度：
- 主族：通道 c 只写 role c；single-random / single-central / multi8 / multi16 / distributed；
- overlap 族：两通道都可写 role0∪role1，每通道 8+8，alpha=0/0.5/1，共享节点符号对半反向；
- 输入预算：‖B_c‖₂ = β = 0.7·√64 = 5.6，|B_ci| = β/√k，固定种子 Rademacher 符号；
- 实现：先按 v15 完整构建模型（RNG 流不变），再用独立 RNG 覆盖 B。
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
BETA = .7 * math.sqrt(M)                     # 5.6，与 v15 期望尺度一致
SEEDS = [11, 22, 33, 44, 55]
EPOCHS = 65
LR = .003
BATCH = 128
THRESHOLD = .9
BUFFER_PER_TASK = 32
ORDERS = {'o0': [0, 1, 2, 3], 'o1': [3, 2, 1, 0], 'o2': [1, 3, 0, 2]}
RATIOS = {0: '0', .125: '12.5'}
MAIN_TOPO = ['single-random', 'single-central', 'multi8', 'multi16', 'distributed']
TOPO_K = {'single-random': 1, 'single-central': 1, 'multi8': 8, 'multi16': 16, 'distributed': 64}
TOPO_OFFSET = {'single-random': 1, 'single-central': 2, 'multi8': 3, 'multi16': 4,
               'distributed': 5, 'overlap': 6}


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
    """W + Route + Hold + Readout；与 v15 相同，W 初始谱范数由 sw 指定。"""

    def __init__(self, sw=.9):
        super().__init__()
        n = N
        self.n = n
        self.m = n // 4
        allowed = torch.tensor([[1, 0, 1, 1], [0, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1.]], dtype=torch.float32)
        mask = (torch.rand(n, n) < .3).float() * allowed.repeat_interleave(self.m, 0).repeat_interleave(self.m, 1)
        w = torch.randn(n, n) * mask
        self.register_buffer('mask', mask)
        self.W = nn.Parameter(sw * w / torch.linalg.matrix_norm(w, 2))
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

    def forward(self, x, meta):
        h = x.new_zeros(len(x), 4, self.m)
        w = (self.W * self.mask).reshape(4, self.m, 4, self.m)
        for t in range(x.shape[1]):
            if self.training and t and t % 5 == 0:
                h = h.detach()
            ctl = meta[:, t]
            a = .2 * self.hold(ctl).sigmoid()
            pooled = h.sum(-1) / self.m
            gate = self.route(torch.cat([ctl, pooled], 1)).sigmoid().reshape(-1, 4, 4)
            rec = (torch.einsum('bsi,sidj->bsdj', h, w) * gate[:, :, :, None]).sum(1)
            cand = torch.tanh(rec + (x[:, t] @ self.B).reshape(-1, 4, self.m))
            h = (1 - a[:, :, None]) * h + a[:, :, None] * cand
        return torch.cat([F.linear(h[:, 2], self.headX.weight, self.headX.bias),
                          F.linear(h[:, 3], self.headY.weight, self.headY.bias)], 1)


def build(seed, sw=.9):
    torch.manual_seed(seed)
    return FlowV2(sw)


def _rademacher(g, k):
    return torch.randint(0, 2, (k,), generator=g).float() * 2 - 1


def make_B(m, seed, topo, alpha=None):
    """独立 RNG 生成 B 并覆盖；返回 spec（节点/符号记录）。"""
    off = TOPO_OFFSET[topo] * 1009 + (0 if alpha is None else int(round(alpha * 10)))
    g = torch.Generator().manual_seed(seed * 104729 + off)
    B = torch.zeros(2, N)
    spec = {'topo': topo, 'beta': BETA, 'alpha': alpha}
    if topo in TOPO_K:
        k = TOPO_K[topo]
        amp = BETA / math.sqrt(k)
        nodes = {}
        for c in range(2):
            if topo == 'single-central':
                Wr = (m.W.detach() * m.mask)[c * M:(c + 1) * M]
                idx = torch.tensor([int(Wr.norm(dim=1).argmax())])
            else:
                idx = torch.randperm(M, generator=g)[:k]
            signs = _rademacher(g, k)
            B[c, c * M + idx] = amp * signs
            nodes[c] = {'idx': idx.tolist(), 'signs': signs.tolist()}
        spec.update({'k': k, 'nodes': nodes, 'amp': amp})
    elif topo == 'overlap':
        per_role = 8
        shared_per_role = int(round(per_role * alpha))
        shift = per_role - shared_per_role
        amp = BETA / math.sqrt(2 * per_role)
        nodes = {'0': [], '1': []}
        for role in (0, 1):
            pool = torch.randperm(M, generator=g)
            ch0 = pool[:per_role]
            ch1 = pool[shift:shift + per_role]
            s0 = _rademacher(g, per_role)
            s1 = _rademacher(g, per_role)
            for i in range(shared_per_role):
                s1[i] = s0[shift + i] * (1 if i % 2 == 0 else -1)
            B[0, role * M + ch0] = amp * s0
            B[1, role * M + ch1] = amp * s1
            nodes['0'].append({'role': role, 'idx': ch0.tolist(), 'signs': s0.tolist()})
            nodes['1'].append({'role': role, 'idx': ch1.tolist(), 'signs': s1.tolist()})
        spec.update({'k': 2 * per_role, 'per_role': per_role, 'shared_per_role': shared_per_role,
                     'nodes': nodes, 'amp': amp})
    else:
        raise ValueError(topo)
    with torch.no_grad():
        m.B.copy_(B)
    return spec


@torch.no_grad()
def saturation_stats(m, seed, task=0, n=256):
    """记录 t=0/1 的输入饱和指标。"""
    x, meta, _ = data(seed * 10000 + task * 100 + 10, n, task)
    out = {}
    h = torch.zeros(n, 4, M)
    w = (m.W * m.mask).reshape(4, M, 4, M)
    for t in range(2):
        z = x[:, t] @ m.B
        active = z.abs().sum(1) > 0
        out[f'sat_prob_t{t}'] = float(((z[active].abs() > 2).float().mean()) if active.any() else 0.0)
        ctl = meta[:, t]
        a = .2 * m.hold(ctl).sigmoid()
        pooled = h.sum(-1) / M
        gate = m.route(torch.cat([ctl, pooled], 1)).sigmoid().reshape(-1, 4, 4)
        rec = (torch.einsum('bsi,sidj->bsdj', h, w) * gate[:, :, :, None]).sum(1)
        cand = torch.tanh(rec + z.reshape(-1, 4, M))
        out[f'mean_tanh_grad_t{t}'] = float((1 - cand ** 2).mean())
        h = (1 - a[:, :, None]) * h + a[:, :, None] * cand
    return out


@torch.no_grad()
def evaluate(m, d):
    m.eval()
    correct = 0
    for ids in torch.arange(len(d[0])).split(128):
        x = d[0][ids]
        meta = d[1][ids]
        y = d[2][ids]
        correct += int(((m(x, meta) > 0) == y.bool()).all(1).sum())
    return correct / len(d[0])


def scores(m, seed):
    return [evaluate(m, data(seed * 10000 + t * 100 + 10, 1024, t)) for t in range(4)]


def train_stage(m, seed, task, epochs, lr, replay_n=0, buffer=None):
    tr = data(seed * 10000 + task * 100, 512, task)
    val = data(seed * 10000 + task * 100 + 1, 256, task)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=lr)
    gen = torch.Generator().manual_seed(seed + task * 100 + 5000)
    gen_rep = torch.Generator().manual_seed(seed + task * 100 + 9000)
    cur_n = BATCH - replay_n
    has_replay = replay_n > 0 and buffer is not None and len(buffer[0]) > 0
    hh = []
    for epoch in range(epochs):
        m.train()
        loss_sum = 0
        order = torch.randperm(512, generator=gen)
        chunks = order[:4 * cur_n].split(cur_n)
        for chunk in chunks:
            if has_replay:
                bx, bm, by = (torch.cat(b) for b in buffer)
                rid = torch.randint(len(bx), (replay_n,), generator=gen_rep)
                xb = torch.cat([tr[0][chunk], bx[rid]])
                mb = torch.cat([tr[1][chunk], bm[rid]])
                yb = torch.cat([tr[2][chunk], by[rid]])
            else:
                xb = tr[0][chunk]
                mb = tr[1][chunk]
                yb = tr[2][chunk]
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(m(xb, mb), yb)
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
    forget = float(np.mean([max(matrix[s][t] for s in range(i, n)) - matrix[-1][t]
                            for i, t in enumerate(order[:-1])]))
    bwt = float(np.mean([matrix[-1][t] - matrix[i][t] for i, t in enumerate(order[:-1])]))
    return acq, final, forget, bwt


def run_cl(topo, order_key, ratio, seed, out, epochs, sw=.9, alpha=None):
    label = RATIOS[ratio]
    atag = '' if alpha is None else f'_a{alpha:g}'
    stag = '' if sw == .9 else f'_s{sw:g}'
    name = f'{topo}{stag}{atag}_{order_key}_r{label}_{seed}'
    p = out / (name + '.json')
    if p.exists():
        return
    order = ORDERS[order_key]
    replay_n = int(round(BATCH * ratio))
    m = build(seed, sw)
    spec = make_B(m, seed, topo, alpha)
    sat = saturation_stats(m, seed)
    buffer = ([], [], [])
    matrix = []
    history = {}
    epochs_to_90 = {}
    start = time.perf_counter()
    for i, task in enumerate(order):
        rn = replay_n if i > 0 else 0
        hh = train_stage(m, seed, task, epochs, LR, rn, buffer)
        history[str(task)] = hh
        epochs_to_90[str(task)] = next((j + 1 for j, h in enumerate(hh) if h['validation'] >= THRESHOLD), None)
        matrix.append(scores(m, seed))
        torch.save({'state': m.state_dict(), 'order': order_key, 'ratio': ratio, 'seed': seed,
                    'stage': i, 'task': task, 'topo': topo, 'alpha': alpha, 'sw': sw},
                   out / f'{name}_stage{i}.pt')
        if ratio > 0:
            tr = data(seed * 10000 + task * 100, 512, task)
            for k in range(3):
                buffer[k].append(tr[k][:BUFFER_PER_TASK])
        print(name, 'stage', i, [round(v, 3) for v in matrix[-1]], flush=True)
    acq, final, forget, bwt = metrics(matrix, order)
    row = {'name': name, 'topo': topo, 'alpha': alpha, 'sw': sw, 'order': order_key, 'order_seq': order,
           'ratio': ratio, 'ratio_label': label, 'replay_n': replay_n, 'seed': seed, 'b_spec': spec,
           'saturation': sat, 'matrix': matrix, 'history': history, 'epochs_to_90': epochs_to_90,
           'acquisition': acq, 'final_mean': final, 'forgetting': forget, 'bwt': bwt,
           'buffer_per_task': BUFFER_PER_TASK, 'params': sum(p.numel() for p in m.parameters() if p.requires_grad),
           'seconds': time.perf_counter() - start, 'threads': torch.get_num_threads(),
           'epochs': epochs, 'lr': LR}
    p.write_text(json.dumps(row, indent=2))


def run_single(topo, task, seed, out, epochs, sw=.9, alpha=None):
    atag = '' if alpha is None else f'_a{alpha:g}'
    stag = '' if sw == .9 else f'_s{sw:g}'
    name = f'single_{topo}{stag}{atag}_t{task}_{seed}'
    p = out / (name + '.json')
    if p.exists():
        return
    m = build(seed, sw)
    spec = make_B(m, seed, topo, alpha)
    sat = saturation_stats(m, seed, task)
    start = time.perf_counter()
    hh = train_stage(m, seed, task, epochs, LR)
    final = evaluate(m, data(seed * 10000 + task * 100 + 10, 1024, task))
    torch.save({'state': m.state_dict(), 'seed': seed, 'task': task, 'topo': topo, 'alpha': alpha, 'sw': sw},
               out / f'{name}.pt')
    row = {'name': name, 'topo': topo, 'alpha': alpha, 'sw': sw, 'task': task, 'seed': seed,
           'b_spec': spec, 'saturation': sat, 'final': final, 'history': hh,
           'seconds': time.perf_counter() - start, 'threads': torch.get_num_threads(),
           'epochs': epochs, 'lr': LR}
    p.write_text(json.dumps(row, indent=2))


def _job(args):
    kind, kw = args
    if kind == 'cl':
        run_cl(**kw)
    else:
        run_single(**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', default='A', choices=['A', 'B', 'C'])
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--orders', nargs='+', default=list(ORDERS))
    ap.add_argument('--ratios', nargs='+', type=float, default=list(RATIOS))
    ap.add_argument('--topos', nargs='+', default=None)
    ap.add_argument('--alphas', nargs='+', type=float, default=[0, .5, 1])
    ap.add_argument('--sws', nargs='+', type=float, default=[.7, .9, 1.1])
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--out', default='results/A')
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--no-single', action='store_true')
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    ratios = [r for r in RATIOS if r in args.ratios]
    topos = args.topos or (MAIN_TOPO if args.phase == 'A' else
                           (['overlap'] if args.phase == 'B' else ['single-random', 'distributed']))
    sws = [.9] if args.phase in ('A', 'B') else args.sws
    alphas = [None] if args.phase in ('A', 'C') else args.alphas
    (out / 'config.json').write_text(json.dumps(
        {'phase': args.phase, 'n': N, 'beta': BETA, 'seeds': SEEDS, 'orders': ORDERS, 'ratios': RATIOS,
         'topos': topos, 'alphas': alphas, 'sws': sws, 'epochs': args.epochs, 'lr': LR, 'batch': BATCH,
         'buffer_per_task': BUFFER_PER_TASK, 'arch': 'Flow-v2 (W+Route+Hold+Readout)'}, indent=2))
    print('phase', args.phase, 'topos', topos, 'sws', sws, 'alphas', alphas,
          'ratios', ratios, 'epochs', args.epochs, flush=True)
    jobs = []
    for topo in topos:
        for alpha in alphas:
            for sw in sws:
                for seed in args.seeds:
                    for order_key in args.orders:
                        for ratio in ratios:
                            jobs.append(('cl', dict(topo=topo, order_key=order_key, ratio=ratio, seed=seed,
                                                    out=out, epochs=args.epochs, sw=sw, alpha=alpha)))
                    if not args.no_single:
                        for task in range(4):
                            jobs.append(('single', dict(topo=topo, task=task, seed=seed, out=out,
                                                        epochs=args.epochs, sw=sw, alpha=alpha)))
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

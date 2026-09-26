"""第十九轮 Phase 0B：Flow-v2 块仿射算子结构全量扫描（无训练，只读 V15/V17 检查点）。

测：
- 解析 J_t = S_t + C_t（C 来自 pooled(h)->Route，rank<=4）与 autograd 的一致性；
- 单步 / 块 Jacobian 谱与显著秩（验证 rank(J-S)<=4L）；
- scan tree 每层显著秩、截断再压缩误差；
- （L=4）PreRoute 中心下 K 遍求解的精度（比较 C 截断 r_max 的影响）。
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
from torch.func import jacrev, vmap

ROOT = Path(__file__).parent
V15 = (ROOT / '../flow_mvp_v15/results').resolve()
V17 = (ROOT / '../flow_mvp_v17/results').resolve()
N = 256
M = N // 4
SEEDS = [11, 22, 33, 44, 55]
TASKS = [0, 1, 2, 3]
LS = [1, 2, 4, 5, 10]
TOLS = [1e-3, 1e-4, 1e-5]
RMAXS = [None, 64, 32, 16, 8, 4]
ALLOWED = torch.tensor([[1, 0, 1, 1], [0, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1.]], dtype=torch.float32)


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
    m = FlowV2()
    ck = torch.load(V15 / f'o0_flowv2_r12.5_{seed}_stage3.pt', map_location='cpu', weights_only=False)
    m.load_state_dict(ck['state'])
    m.eval()
    return m


def load_preroute(seed):
    m = PreRoute()
    ck = torch.load(V17 / f'o0_preroute_r12.5_{seed}_stage3.pt', map_location='cpu', weights_only=False)
    m.load_state_dict(ck['state'])
    m.eval()
    return m


def readout(m, h):
    return torch.cat([torch.nn.functional.linear(h[:, 2], m.headX.weight, m.headX.bias),
                      torch.nn.functional.linear(h[:, 3], m.headY.weight, m.headY.bias)], 1)


def step_map(m, h, xt, mt):
    w = (m.W * m.mask).reshape(4, M, 4, M)
    a = .2 * m.hold(mt).sigmoid()
    gate = m.route(torch.cat([mt, h.sum(-1) / M], 1)).sigmoid().reshape(-1, 4, 4)
    rec = (torch.einsum('bsi,sidj->bsdj', h, w) * gate[:, :, :, None]).sum(1)
    cand = torch.tanh(rec + (xt @ m.B).reshape(-1, 4, M))
    h_new = (1 - a[:, :, None]) * h + a[:, :, None] * cand
    return h_new, cand, gate, a


def block_map(m, h, xb, mb):
    for t in range(xb.shape[1]):
        h, _, _, _ = step_map(m, h, xb[:, t], mb[:, t])
    return h


def serial_traj(m, x, meta):
    h = x.new_zeros(len(x), 4, M)
    hs = [h]
    for t in range(x.shape[1]):
        h, _, _, _ = step_map(m, h, x[:, t], meta[:, t])
        hs.append(h)
    return hs


def route_jac(m, mt, pooled):
    def f(mtt, p):
        return m.route(torch.cat([mtt, p], 0)).sigmoid()

    with torch.enable_grad():
        return vmap(jacrev(f, argnums=1), in_dims=(0, 0))(mt, pooled)


def step_operator(m, h, xt, mt):
    """解析 J = S + C（rank(C)<=4）。返回 h_new [b,4,M], S,C [b,N,N], (cand,gate,a)。"""
    b = len(h)
    w = (m.W * m.mask).reshape(4, M, 4, M)
    h_new, cand, gate, a = step_map(m, h, xt, mt)
    A = (gate[:, :, None, :, None] * w[None]).permute(0, 3, 4, 1, 2).reshape(b, N, N)
    a_rep = a.repeat_interleave(M, dim=1)
    da = a_rep * (1 - cand.reshape(b, N) ** 2)
    eye = torch.eye(N)
    S = (1 - a_rep)[:, :, None] * eye + da[:, :, None] * A
    pooled = h.sum(-1) / M
    G = route_jac(m, mt, pooled)                # [b,16,4] flat=(s*4+d) of d sigmoid(route)/d pooled
    Grows = G.reshape(b, 4, 4, 4).permute(0, 2, 1, 3).reshape(b, 16, 4)  # row=(d*4+p), col=s
    R = torch.einsum('bsi,sidj->bsdj', h, w)    # [b, s', d, j]
    X = R.permute(0, 2, 3, 1)                   # [b,d,j,s']
    M1 = torch.zeros(b, N, 16)
    for d in range(4):
        M1[:, d * M:(d + 1) * M, d * 4:(d + 1) * 4] = X[:, d]
    Gp = Grows.repeat_interleave(M, dim=2) / M  # [b,16,256]: Gp[(d,p),(s,i)] = G[(p,d),s]/M
    C = da[:, :, None] * (M1 @ Gp)
    return h_new, S, C, (cand, gate, a)


def block_operator(m, h_entry, xb, mb):
    """块算子 (S,C,d)：linearized at block entry. exit = F(h_entry)."""
    b = len(h_entry)
    S = torch.eye(N).expand(b, N, N).clone()
    C = torch.zeros(b, N, N)
    h = h_entry
    for t in range(xb.shape[1]):
        h, S_t, C_t, _ = step_operator(m, h, xb[:, t], mb[:, t])
        C = S_t @ C + C_t @ S + C_t @ C
        S = S_t @ S
    M_full = S + C
    d = h.reshape(b, N) - (M_full @ h_entry.reshape(b, N)[..., None])[..., 0]
    return S, C, d


def sig_rank(sv, tol=1e-4):
    return int((sv > tol * sv[0]).sum().item()) if len(sv) else 0


def truncate_C(C, rmax):
    if rmax is None:
        return C
    U, s, Vh = torch.linalg.svd(C, full_matrices=False)
    r = min(rmax, s.shape[-1])
    return (U[:, :, :r] * s[:, None, :r]) @ Vh[:, :r, :]


def check_support(S):
    # S 的行块=目的角色 d，列块=来源角色 s；ALLOWED 以 [s,d] 索引，故比较转置
    blocks = (S.reshape(4, M, 4, M).abs().amax(dim=(1, 3)) > 1e-9).float()
    return int((blocks - ALLOWED.T).abs().sum().item())


def eh(est, true):
    vals = []
    for a, b in zip(est[1:], true[1:]):
        vals.append(float((a - b).norm(dim=(1, 2)).mean() / (b.norm(dim=(1, 2)).mean() + 1e-9)))
    return float(np.mean(vals))


def acc_from_state(m, h, y):
    return float(((readout(m, h) > 0) == y.bool()).all(1).float().mean())


def numeric_step_jac(m, h, xt, mt):
    def f(flat):
        out, _, _, _ = step_map(m, flat.reshape(1, 4, M), xt.reshape(1, 2), mt.reshape(1, 7))
        return out.reshape(-1)

    with torch.enable_grad():
        return jacrev(f)(h.reshape(-1))


def numeric_block_jac(m, h, xb, mb):
    def f(flat):
        return block_map(m, flat.reshape(1, 4, M), xb.reshape(1, xb.shape[0], 2), mb.reshape(1, mb.shape[0], 7)).reshape(-1)

    with torch.enable_grad():
        return jacrev(f)(h.reshape(-1))


def run_solver_chain(fm, x, meta, y, p0, B, L, rmax=None, npasses=4):
    """在给定中心初值 p0 上做 K 遍 relinearization；返回 {K: {acc, Eh}}。"""
    with torch.no_grad():
        true = serial_traj(fm, x, meta)
        true_bounds = [true[b * L] for b in range(B + 1)]
        p = [t.clone() for t in p0]
        chain = {}
        for K in range(npasses):
            bounds = [torch.zeros(len(x), N)]
            h = bounds[0]
            for b in range(B):
                xb = x[:, b * L:(b + 1) * L]
                mb = meta[:, b * L:(b + 1) * L]
                pb4 = p[b].reshape(len(x), 4, M)
                Fb = block_map(fm, pb4, xb, mb).reshape(len(x), N)
                S, C, _ = block_operator(fm, pb4, xb, mb)
                Mop = S + truncate_C(C, rmax)
                h = Fb + (Mop @ (h - p[b])[..., None])[..., 0]
                bounds.append(h)
            p = [t.detach() for t in bounds]
            d_abs = [float((p[i] - true_bounds[i].reshape(len(x), N)).norm(dim=1).mean()) for i in range(1, B + 1)]
            t_abs = [float(true_bounds[i].reshape(len(x), N).norm(dim=1).mean()) for i in range(1, B + 1)]
            chain[K] = {'acc': acc_from_state(fm, p[-1].reshape(len(x), 4, M), y),
                        'Eh': eh([t.reshape(len(x), 4, M) for t in p],
                                 [t.detach() for t in true_bounds]),
                        'Eh_abs': float(np.mean(d_abs)),
                        'Eh_glob': float(sum(d_abs) / (sum(t_abs) + 1e-12))}
    return chain


def run_job(seed, task, steps, samples, ls, do_side=True, do_identity=True):
    t0 = time.perf_counter()
    fm = load_flow(seed)
    pm = load_preroute(seed)
    x, meta, y = data(seed * 10000 + task * 100 + 10, samples, task, steps)
    out = {'seed': seed, 'task': task, 'steps': steps, 'samples': samples, 'ls': ls}

    with torch.no_grad():
        true = serial_traj(fm, x, meta)

        # ---- single-step spectra ----
        cs, js, jmi = [], [], []
        for t in range(steps):
            _, S, C, _ = step_operator(fm, true[t], x[:, t], meta[:, t])
            cs.append(torch.linalg.svdvals(C))
            J = S + C
            js.append(torch.linalg.svdvals(J)[:, 0])
            jmi.append(torch.linalg.svdvals(J - torch.eye(N))[:, 0])
        cs = torch.stack(cs)                       # [T,b,N]
        out['step'] = {
            'C_sv_mean': [float(v) for v in cs.mean(dim=(0, 1))[:8]],
            'C_sv0_mean': float(cs[..., 0].mean()),
            'C_sig_rank_mean': {f'{tol:g}': float(np.mean([sig_rank(cs[t, i], tol) for t in range(steps) for i in range(samples)])) for tol in TOLS},
            'J_sv0_mean': float(torch.stack(js).mean()),
            'JmI_sv0_mean': float(torch.stack(jmi).mean()),
        }

        # ---- identity check vs autograd ----
        if do_identity:
            errs_step, errs_block = [], []
            for t in [1, 3, min(7, steps - 1)]:
                _, S, C, _ = step_operator(fm, true[t], x[:, t], meta[:, t])
                Jn = numeric_step_jac(fm, true[t][0], x[0, t], meta[0, t])
                Ja = (S + C)[0]
                errs_step.append(float((Jn - Ja).norm() / (Ja.norm() + 1e-12)))
            L = 4 if steps % 4 == 0 else 2
            Sblk, Cblk, _ = block_operator(fm, true[0], x[:, :L], meta[:, :L])
            Jn = numeric_block_jac(fm, true[0][0], x[0, :L], meta[0, :L])
            Ja = (Sblk + Cblk)[0]
            errs_block.append(float((Jn - Ja).norm() / (Ja.norm() + 1e-12)))
            out['identity'] = {'step_rel_err_max': max(errs_step), 'block_rel_err': max(errs_block), 'L_block': L}

        # ---- per-block operators & tree ----
        out['block'], out['tree'] = {}, {}
        block_ops = {}
        for L in ls:
            if steps % L:
                continue
            B = steps // L
            S_list, C_list, d_exact = [], [], []
            rank_by_tol = {f'{tol:g}': [] for tol in TOLS}
            sup = 0
            for b in range(B):
                e = true[b * L]
                S, C, d = block_operator(fm, e, x[:, b * L:(b + 1) * L], meta[:, b * L:(b + 1) * L])
                S_list.append(S)
                C_list.append(C)
                d_exact.append(d)
                sup += sum(check_support(S[i]) for i in range(samples))
                sv = torch.linalg.svdvals(C)
                for tol in TOLS:
                    rank_by_tol[f'{tol:g}'].append(np.mean([sig_rank(sv[i], tol) for i in range(samples)]))
            block_ops[L] = (S_list, C_list, d_exact)
            out['block'][str(L)] = {
                'C_sv0_mean': float(np.mean([float(torch.linalg.svdvals(C_list[b])[i, 0]) for b in range(B) for i in range(samples)])),
                'C_sig_rank_mean': {k: float(np.mean(v)) for k, v in rank_by_tol.items()},
                'C_sig_rank_max': {k: float(np.max(v)) for k, v in rank_by_tol.items()},
                'support_violations': sup,
            }

            # exact tree
            nodes = [(S_list[b], C_list[b], d_exact[b]) for b in range(B)]
            levels = []
            while len(nodes) > 1:
                nxt = []
                for j in range(0, len(nodes) - 1, 2):
                    S1, C1, d1 = nodes[j]
                    S2, C2, d2 = nodes[j + 1]
                    S = S2 @ S1
                    C = S2 @ C1 + C2 @ S1 + C2 @ C1
                    d = (S2 + C2) @ d1[..., None]
                    d = d[..., 0] + d2
                    nxt.append((S, C, d))
                if len(nodes) % 2:
                    nxt.append(nodes[-1])
                nodes = nxt
                sv = torch.linalg.svdvals(nodes[0][1])
                levels.append({
                    'nodes': len(nodes),
                    'sig_rank': float(np.mean([sig_rank(sv[i]) for i in range(samples)])),
                    'sv0': float(sv[:, 0].mean()),
                    'tail_ratio': float((sv[:, 16:].max() / (sv[:, 0] + 1e-12)).mean()) if sv.shape[1] > 16 else 0.0,
                })
            S_root, C_root, _ = nodes[0]

            # truncated tree
            trunc = {}
            for rmax in RMAXS:
                nodes_t = []
                for b in range(B):
                    nodes_t.append((S_list[b], truncate_C(C_list[b], rmax), d_exact[b]))
                while len(nodes_t) > 1:
                    nxt = []
                    for j in range(0, len(nodes_t) - 1, 2):
                        S1, C1, d1 = nodes_t[j]
                        S2, C2, d2 = nodes_t[j + 1]
                        S = S2 @ S1
                        C = truncate_C(S2 @ C1 + C2 @ S1 + C2 @ C1, rmax)
                        d = (S2 + C2) @ d1[..., None]
                        d = d[..., 0] + d2
                        nxt.append((S, C, d))
                    if len(nodes_t) % 2:
                        nxt.append(nodes_t[-1])
                    nodes_t = nxt
                Ct = nodes_t[0][1]
                dC = Ct - C_root
                err_rel = float(dC.norm() / (C_root.norm() + 1e-12))
                v = torch.randn(N, samples, generator=torch.Generator().manual_seed(0))
                v = v / (v.norm(dim=0, keepdim=True) + 1e-12)
                vp = v.unsqueeze(0).expand(len(dC), N, samples)
                probe = float((dC @ vp).norm() / ((C_root @ vp).norm() + 1e-12))
                trunc['full' if rmax is None else str(rmax)] = {'C_rel_err': err_rel, 'probe_rel_err': probe}
            out['tree'][str(L)] = {'levels': levels, 'trunc': trunc}

        # ---- side test: L=4, 三种中心初值 x K 遍 x C 截断 ----
        if do_side and steps % 4 == 0:
            L = 4
            B = steps // L
            with torch.no_grad():
                _, ptraj = pm(x, meta, record=True)
            inits = {'pre': [torch.zeros(len(x), N)]}
            for b in range(1, B):
                inits['pre'].append(ptraj[:, b * L - 1].reshape(len(x), N))
            inits['zero'] = [torch.zeros(len(x), N) for _ in range(B)]
            gen = torch.Generator().manual_seed(seed * 100 + task)
            for sigma in (0.05, 0.2):
                pin = [true[b * L].reshape(len(x), N) + sigma * torch.randn(len(x), N, generator=gen)
                       for b in range(B)]
                pin[0] = torch.zeros(len(x), N)
                inits[f'noise{sigma:g}'] = pin
            side = {}
            for iname, p0 in inits.items():
                rows = {}
                for rmax in RMAXS:
                    key = 'full' if rmax is None else str(rmax)
                    chain = run_solver_chain(fm, x, meta, y, p0, B, L, rmax=rmax)
                    rows[key] = {str(K): v for K, v in chain.items()}
                side[iname] = rows
            out['side'] = {'L': L, 'inits': side}

    out['seconds'] = time.perf_counter() - t0
    return out


def run_job_star(args):
    torch.set_num_threads(int(os.environ.get('FLOW_THREADS', '1')))
    return run_job(*args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--tasks', nargs='+', type=int, default=TASKS)
    ap.add_argument('--steps', nargs='+', type=int, default=[20, 40])
    ap.add_argument('--ls', nargs='+', type=int, default=LS)
    ap.add_argument('--samples', type=int, default=4)
    ap.add_argument('--jobs', type=int, default=12)
    ap.add_argument('--no-side', action='store_true')
    ap.add_argument('--no-identity', action='store_true')
    ap.add_argument('--out', default='results/raw')
    args = ap.parse_args()

    torch.use_deterministic_algorithms(True)
    outdir = ROOT / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = [(s, t, st, args.samples, args.ls, not args.no_side, not args.no_identity)
            for s in args.seeds for t in args.tasks for st in args.steps]
    t0 = time.perf_counter()
    if args.jobs <= 1:
        for job in jobs:
            r = run_job(*job)
            (outdir / f'{r["seed"]}_{r["task"]}_{r["steps"]}.json').write_text(json.dumps(r))
            print('done', r['seed'], r['task'], r['steps'], round(r['seconds'], 1), 's', flush=True)
    else:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(args.jobs) as pool:
            for r in pool.imap_unordered(run_job_star, jobs):
                (outdir / f'{r["seed"]}_{r["task"]}_{r["steps"]}.json').write_text(json.dumps(r))
                print('done', r['seed'], r['task'], r['steps'], round(r['seconds'], 1), 's', flush=True)
    print('all', len(jobs), 'jobs in', round(time.perf_counter() - t0, 1), 's')


if __name__ == '__main__':
    main()

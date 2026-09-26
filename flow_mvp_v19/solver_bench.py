"""第十九轮 Phase 0C：GPU batch 化 solver 与计时。

实现：
- structured：块算子 S + UVᵀ（解析构造、每块压缩到 r_max），Hillis-Steele prefix scan，K 遍重线性化；
- gs_jvp：V18 原路线（逐块 jvp，Gauss-Seidel 方向）；
- serial：Flow-v2 串行前向。
指标：accuracy、latency（warmup + cuda synchronize）、peak memory、operator bytes、有效深度。
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import experiment as E

N, M = E.N, E.M


def route_weights(m):
    return m.route[0].weight, m.route[0].bias, m.route[2].weight, m.route[2].bias


def route_jac_analytic(m, q):
    W1, b1, W2, b2 = route_weights(m)
    h1 = torch.tanh(q @ W1.t() + b1)
    z2 = h1 @ W2.t() + b2
    g = torch.sigmoid(z2)
    W1p = W1[:, 7:]
    J = (g * (1 - g)).unsqueeze(-1) * (W2 @ ((1 - h1 ** 2).unsqueeze(-1) * W1p))
    return g, J


def init_S(P, dev):
    S = torch.zeros(P, 4, 4, M, M, device=dev)
    eye = torch.eye(M, device=dev)
    for d in range(4):
        S[:, d, d] = eye
    return S


def apply_S(S, X):
    """S [P,4,4,M,M]（dest,src,out,in）作用到 X [P,4,M,r] -> [P,4,M,r]。"""
    out = torch.einsum('pdsji,psir->pdjr', S, X)
    return out


def apply_S_t(S, X):
    """Sᵀ 作用到 X [P,4,M,r]（dest,src,out,in 转置）-> [P,4,M,r]。"""
    return torch.einsum('pdsji,pdjr->psir', S, X)


def compose_S(S2, S1):
    return torch.einsum('pdmjk,pmski->pdsji', S2, S1)


def compress(U, V, rmax, oversample=8):
    """UVᵀ 截断到 rmax：randomized SVD（QR + 小 Gram），避开 cuSOLVER 小矩阵慢路径。"""
    r = U.shape[-1]
    if r > rmax:
        k = min(r, rmax + oversample)
        gen = torch.Generator(device=U.device).manual_seed(1234)
        Om = torch.randn(U.shape[0], N, k, generator=gen, device=U.device, dtype=U.dtype)
        Y = U @ (V.transpose(1, 2) @ Om)                  # C @ Om
        Q, _ = torch.linalg.qr(Y)
        B = (Q.transpose(1, 2) @ U) @ V.transpose(1, 2)   # Qᵀ C, [P,k,N]
        G = B @ B.transpose(1, 2)
        G = G + 1e-8 * torch.eye(k, device=G.device, dtype=G.dtype)[None]
        lam, A = torch.linalg.eigh(G)
        idx = torch.argsort(lam, dim=1, descending=True)[:, :rmax]
        lam = torch.gather(lam, 1, idx).clamp_min(1e-12)
        A = torch.gather(A, 2, idx[:, None, :].expand(-1, A.shape[1], -1))
        s = lam.sqrt()
        U = (Q @ A) * s[:, None, :]
        Bt = B.transpose(1, 2)                            # [P,N,k]
        V = (Bt @ A) / s[:, None, :]
        r = rmax
    if r < rmax:
        P = U.shape[0]
        U = torch.cat([U, U.new_zeros((P, N, rmax - r))], -1)
        V = torch.cat([V, V.new_zeros((P, N, rmax - r))], -1)
    return U.contiguous(), V.contiguous()


def build_block_ops(m, p, xs, ms, rmax=16):
    """p [P,4,M]; xs [P,L,2]; ms [P,L,7] -> S [P,4,4,M,M], U,V [P,N,r], exit [P,4,M]。"""
    P, L = xs.shape[0], xs.shape[1]
    dev = p.device
    W4 = (m.W * m.mask).reshape(4, M, 4, M)
    S = init_S(P, dev)
    U = torch.zeros(P, N, 0, device=dev)
    V = torch.zeros(P, N, 0, device=dev)
    h = p
    Wt = W4.permute(2, 3, 0, 1)                                 # [d,j,s,i]
    for t in range(L):
        a = .2 * m.hold(ms[:, t]).sigmoid()
        pooled = h.sum(-1) / M
        q = torch.cat([ms[:, t], pooled], 1)
        g_flat, Jg = route_jac_analytic(m, q)
        gate = g_flat.reshape(P, 4, 4)
        R = torch.einsum('psi,sidj->psdj', h, W4)                # [P,s,d,j] 未加门
        rec = (R * gate[:, :, :, None]).sum(1)
        cand = torch.tanh(rec + (xs[:, t] @ m.B).reshape(P, 4, M))
        h_new = (1 - a[:, :, None]) * h + a[:, :, None] * cand
        # S_t = diag(1-a) + D_a D_tanh' A(g)
        Ablk = torch.einsum('psd,djsi->pdsji', gate, Wt)
        St = torch.zeros_like(S)
        for d in range(4):
            St[:, d, d] += (1 - a[:, d])[:, None, None] * torch.eye(M, device=dev)[None]
        da_dt = a[:, :, None] * (1 - cand ** 2)                 # [P,4,M]
        St = St + torch.einsum('pdj,pdsji->pdsji', da_dt, Ablk)
        # C_t = U_t V_t^T
        X = R.permute(0, 2, 3, 1)                               # [P,d,j,s']
        M1 = torch.zeros(P, N, 16, device=dev)
        for d in range(4):
            M1[:, d * M:(d + 1) * M, d * 4:(d + 1) * 4] = X[:, d]
        Grows = Jg.reshape(P, 4, 4, 4).permute(0, 2, 1, 3).reshape(P, 16, 4)
        Gp = Grows.repeat_interleave(M, dim=2) / M              # [P,16,N]
        da_dt = (a[:, :, None] * (1 - cand ** 2)).reshape(P, N)
        U_t = da_dt[:, :, None] * M1
        V_t = Gp.transpose(1, 2).contiguous()
        # 累积 (S_t + C_t)(S + C)
        if U.shape[-1] == 0:
            U, V = U_t, V_t
        else:
            SU = apply_S(St, U.reshape(P, 4, M, -1)).reshape(P, N, -1)
            StV = apply_S_t(S, V_t.reshape(P, 4, M, -1)).reshape(P, N, -1)
            UV = torch.bmm(U.transpose(1, 2), V_t)
            Vnew = torch.cat([V, StV + V @ UV], -1)
            U = torch.cat([SU, U_t], -1)
            V = Vnew
        S = compose_S(St, S)
        h = h_new
    U, V = compress(U, V, rmax)
    return S, U, V, h


def apply_op(S, U, V, d, h):
    """(S + UVᵀ) h + d；h [P,4,M] -> [P,4,M]。"""
    P = h.shape[0]
    out = apply_S(S, h.reshape(P, 4, M, 1)).reshape(P, 4, M)
    if U.shape[-1]:
        out = out + (U @ (V.transpose(1, 2) @ h.reshape(P, N, 1))).reshape(P, 4, M)
    return out + d.reshape(P, 4, M)


def compose_ops(A, B, rmax):
    """A∘B（先 B 后 A）：A,B = (S,U,V,d) [P,...]。"""
    S1, U1, V1, d1 = B
    S2, U2, V2, d2 = A
    P = S1.shape[0]
    S = compose_S(S2, S1)
    SU1 = apply_S(S2, U1.reshape(P, 4, M, -1)).reshape(P, N, -1)
    StV2 = apply_S_t(S1, V2.reshape(P, 4, M, -1)).reshape(P, N, -1)
    UV = torch.bmm(U1.transpose(1, 2), V2)
    U = torch.cat([SU1, U2], -1)
    V = torch.cat([V1, StV2 + V1 @ UV], -1)
    U, V = compress(U, V, rmax)
    M2h = apply_S(S2, d1.reshape(P, 4, M, 1)).reshape(P, N)
    if U2.shape[-1]:
        M2h = M2h + U2 @ (V2.transpose(1, 2) @ d1.reshape(P, N, 1))[:, :, 0]
    d = M2h + d2
    return S, U, V, d


def scan_prefix(ops, rmax):
    """Hillis-Steele 前缀复合：op[i] <- op[i]∘...∘op[0]；ops 元组各 [b,B,...]。"""
    S, U, V, d = ops
    b, B = S.shape[0], S.shape[1]
    step = 1
    while step < B:
        S2, U2, V2, d2 = S[:, step:], U[:, step:], V[:, step:], d[:, step:]
        S1, U1, V1, d1 = S[:, :-step], U[:, :-step], V[:, :-step], d[:, :-step]
        n = S2.shape[1]
        s2 = S2.reshape(-1, 4, 4, M, M)
        s1 = S1.reshape(-1, 4, 4, M, M)
        u1 = U1.reshape(-1, N, U1.shape[-1])
        v2 = V2.reshape(-1, N, V2.shape[-1])
        Snew = compose_S(s2, s1).reshape(b, n, 4, 4, M, M)
        SU1 = apply_S(s2, u1.reshape(-1, 4, M, u1.shape[-1])).reshape(-1, N, u1.shape[-1])
        StV2 = apply_S_t(s1, v2.reshape(-1, 4, M, v2.shape[-1])).reshape(-1, N, v2.shape[-1])
        UV = torch.bmm(u1.transpose(1, 2), v2)
        Unew = torch.cat([SU1, U2.reshape(-1, N, U2.shape[-1])], -1)
        Vnew = torch.cat([V1.reshape(-1, N, V1.shape[-1]), StV2 + V1.reshape(-1, N, V1.shape[-1]) @ UV], -1)
        if not (torch.isfinite(Snew).all() and torch.isfinite(Unew).all() and torch.isfinite(Vnew).all()):
            raise OverflowError('scan composition overflow (fp32)')
        Unew, Vnew = compress(Unew, Vnew, rmax)
        M2d1 = apply_S(s2, d1.reshape(-1, 4, M, 1)).reshape(-1, N)
        if U2.shape[-1]:
            M2d1 = M2d1 + (U2.reshape(-1, N, U2.shape[-1])
                           @ (v2.transpose(1, 2) @ d1.reshape(-1, N, 1)))[:, :, 0]
        dnew = M2d1 + d2.reshape(-1, N)
        S = torch.cat([S[:, :step], Snew], 1)
        U = torch.cat([U[:, :step], Unew.reshape(b, n, N, -1)], 1)
        V = torch.cat([V[:, :step], Vnew.reshape(b, n, N, -1)], 1)
        d = torch.cat([d[:, :step], dnew.reshape(b, n, N)], 1)
        step *= 2
    return S, U, V, d


def structured_solve(m, x, meta, L, K, rmax=16, p0=None):
    """返回 boundaries [b,B+1,4,M] 与统计。"""
    b, steps = x.shape[0], x.shape[1]
    B = steps // L
    dev = x.device
    with torch.no_grad():
        p = torch.zeros(b, B, 4, M, device=dev) if p0 is None else p0
        stats = {'ops_bytes': 0, 'compose_calls': 0}
        for k in range(K + 1):
            xs = x.reshape(b, B, L, 2).reshape(b * B, L, 2)
            ms = meta.reshape(b, B, L, 7).reshape(b * B, L, 7)
            ps = p.reshape(b * B, 4, M)
            S, U, V, exit = build_block_ops(m, ps, xs, ms, rmax)
            Mop = apply_S(S, ps.reshape(-1, 4, M, 1)).reshape(-1, N)
            if U.shape[-1]:
                Mop = Mop + (U @ (V.transpose(1, 2) @ ps.reshape(-1, N, 1)))[:, :, 0]
            d = exit.reshape(-1, N) - Mop
            ops = (S.reshape(b, B, 4, 4, M, M), U.reshape(b, B, N, -1),
                   V.reshape(b, B, N, -1), d.reshape(b, B, N))
            stats['ops_bytes'] = int(S.numel() * 4 + U.numel() * 4 + V.numel() * 4 + d.numel() * 4)
            Ps, Pu, Pv, Pd = scan_prefix(ops, rmax)
            h0 = torch.zeros(b, 4, M, device=dev)
            P = b * B
            h_flat = h0.reshape(b, 1, N).expand(b, B, N).reshape(P, N)
            pu = Pu.reshape(P, N, -1)
            pv = Pv.reshape(P, N, -1)
            pd = Pd.reshape(P, N)
            out = apply_S(Ps.reshape(P, 4, 4, M, M), h_flat.reshape(P, 4, M, 1)).reshape(P, N)
            if pu.shape[-1]:
                out = out + (pu @ (pv.transpose(1, 2) @ h_flat.reshape(P, N, 1)))[:, :, 0]
            out = out + pd
            bounds = torch.cat([h0.reshape(b, 1, N), out.reshape(b, B, N)], 1)
            bounds = bounds.reshape(b, B + 1, 4, M)
            p = bounds[:, :B].contiguous()
        return bounds, stats


def structured_solve_chunked(m, x, meta, L, K, rmax, p0):
    b = x.shape[0]
    B = x.shape[1] // L
    chunk = max(1, 256 // B)
    outs, stats = [], {'ops_bytes': 0, 'compose_calls': 0}
    for s in range(0, b, chunk):
        sl = slice(s, min(s + chunk, b))
        bounds, st = structured_solve(m, x[sl], meta[sl], L, K, rmax, None if p0 is None else p0[sl])
        outs.append(bounds)
        stats['ops_bytes'] = max(stats['ops_bytes'], st['ops_bytes'])
    return torch.cat(outs, 0), stats


def structured_seq_solve(m, x, meta, L, K, rmax, p0=None):
    """同一算子，但块间用顺序 apply（不做 scan）。"""
    b, steps = x.shape[0], x.shape[1]
    B = steps // L
    dev = x.device
    with torch.no_grad():
        p = torch.zeros(b, B, 4, M, device=dev) if p0 is None else p0
        for k in range(K + 1):
            xs = x.reshape(b, B, L, 2).reshape(b * B, L, 2)
            ms = meta.reshape(b, B, L, 7).reshape(b * B, L, 7)
            S, U, V, exit_ = build_block_ops(m, p.reshape(b * B, 4, M), xs, ms, rmax)
            Mop = apply_S(S, p.reshape(-1, 4, M, 1)).reshape(-1, N)
            if U.shape[-1]:
                Mop = Mop + (U @ (V.transpose(1, 2) @ p.reshape(-1, N, 1)))[:, :, 0]
            d = (exit_.reshape(-1, N) - Mop).reshape(b, B, N)
            S = S.reshape(b, B, 4, 4, M, M)
            U = U.reshape(b, B, N, -1)
            V = V.reshape(b, B, N, -1)
            h = torch.zeros(b, 4, M, device=dev)
            bounds = [h]
            for i in range(B):
                h = apply_op(S[:, i], U[:, i], V[:, i], d[:, i], h)
                bounds.append(h)
            bounds = torch.stack(bounds, 1)
            p = bounds[:, :B].contiguous()
        return bounds, {'ops_bytes': int(S.numel() * 4)}


def structured_seq_solve_chunked(m, x, meta, L, K, rmax, p0):
    b = x.shape[0]
    B = x.shape[1] // L
    chunk = max(1, 256 // B)
    outs = []
    for s in range(0, b, chunk):
        sl = slice(s, min(s + chunk, b))
        bounds, _ = structured_seq_solve(m, x[sl], meta[sl], L, K, rmax, None if p0 is None else p0[sl])
        outs.append(bounds)
    return torch.cat(outs, 0), {'ops_bytes': 0}


def gs_jvp_solve(m, x, meta, L, K, p0):
    """V18 路线：逐块 jvp，Gauss-Seidel 方向。"""
    from torch.func import jvp
    b, steps = x.shape[0], x.shape[1]
    B = steps // L
    dev = x.device
    p = p0
    with torch.no_grad():
        for k in range(K + 1):
            bounds = [torch.zeros(b, 4, M, device=dev)]
            h = bounds[0]
            for i in range(B):
                xb = x[:, i * L:(i + 1) * L]
                mb = meta[:, i * L:(i + 1) * L]
                f = lambda hh: E.block_map(m, hh, xb, mb)
                ci = f(p[:, i])
                _, jv = jvp(f, (p[:, i],), (h - p[:, i],))
                h = ci + jv
                bounds.append(h)
            p = torch.stack(bounds, 1)[:, :B].contiguous()
        return torch.stack(bounds, 1)


def serial_forward(m, x, meta):
    b, steps = x.shape[0], x.shape[1]
    h = torch.zeros(b, 4, M, device=x.device)
    with torch.no_grad():
        for t in range(steps):
            h, _, _, _ = E.step_map(m, h, x[:, t], meta[:, t])
    return h


def preroute_p0(pm, x, meta, L):
    b, steps = x.shape[0], x.shape[1]
    B = steps // L
    with torch.no_grad():
        _, traj = pm(x, meta, record=True)
    zeros = torch.zeros(b, 4, M, device=x.device)
    p = torch.stack([traj[:, i * L - 1] if i > 0 else zeros for i in range(B)], 1)
    return p


def load_models(seed, dev):
    fm = E.load_flow(seed).to(dev)
    pm = E.load_preroute(seed).to(dev)
    return fm, pm


def accuracy(fm, bounds_or_h, y):
    if bounds_or_h.dim() == 4:
        h = bounds_or_h[:, -1]
    else:
        h = bounds_or_h
    return E.acc_from_state(fm, h, y)


def bench_case(args, seed, task, steps, batch, L, K, rmax, dev):
    fm, pm = load_models(seed, dev)
    B = steps // L
    x, meta, y = E.data(seed * 10000 + task * 100 + 10, 256, task, steps)
    x, meta, y = x.to(dev), meta.to(dev), y.to(dev)
    out = {'seed': seed, 'task': task, 'steps': steps, 'batch': batch, 'L': L, 'K': K,
           'rmax': rmax, 'device': str(dev)}

    def timeit(fn, warmup=3, iters=10):
        for _ in range(warmup):
            fn()
        if dev.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        peak = torch.cuda.max_memory_allocated() if dev.type == 'cuda' else 0
        return float(np.median(ts)), float(peak)

    xb, mb, yb = x[:batch], meta[:batch], y[:batch]
    if getattr(args, 'init', 'zero') == 'preroute':
        p_full = preroute_p0(pm, x, meta, L)
    else:
        p_full = torch.zeros(x.shape[0], B, 4, M, device=dev)
    p0 = p_full[:batch]
    variants = {
        'serial': lambda: serial_forward(fm, xb, mb),
        'gs_jvp': lambda: gs_jvp_solve(fm, xb, mb, L, K, p0),
        'structured': lambda: structured_solve_chunked(fm, xb, mb, L, K, rmax, p0),
        'structured_seq': lambda: structured_seq_solve_chunked(fm, xb, mb, L, K, rmax, p0),
    }
    acc_fn = {
        'serial': lambda: serial_forward(fm, x, meta),
        'gs_jvp': lambda: gs_jvp_solve(fm, x, meta, L, K, p_full),
        'structured': lambda: structured_solve_chunked(fm, x, meta, L, K, rmax, p_full),
        'structured_seq': lambda: structured_seq_solve_chunked(fm, x, meta, L, K, rmax, p_full),
    }
    for name, fn in variants.items():
        try:
            lat, peak = timeit(fn)
            out[name] = {'latency_s': lat, 'peak_bytes': peak}
        except (OverflowError, RuntimeError, torch._C._LinAlgError) as e:
            out[name] = {'error': type(e).__name__ + ': ' + str(e)[:120]}
            continue
        try:
            with torch.no_grad():
                res = acc_fn[name]()
                bounds = res[0] if isinstance(res, tuple) else res
                out[name]['acc256'] = accuracy(fm, bounds, y)
                if isinstance(res, tuple) and len(res) > 1 and isinstance(res[1], dict):
                    out[name].update(res[1])
        except (OverflowError, RuntimeError, torch._C._LinAlgError) as e:
            out[name]['acc_error'] = type(e).__name__
    out['effective_depth_scan'] = (K + 1) * L + (B - 1).bit_length()
    out['effective_depth_seq'] = (K + 1) * (L + x.shape[1] // L)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--task', type=int, default=0)
    ap.add_argument('--steps', nargs='+', type=int, default=[40, 128, 512])
    ap.add_argument('--batch', nargs='+', type=int, default=[1, 8, 32])
    ap.add_argument('--ls', nargs='+', type=int, default=[4])
    ap.add_argument('--ks', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--rmax', type=int, default=16)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default='results/solver_bench.json')
    ap.add_argument('--init', default='zero', choices=['zero', 'preroute'])
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    if args.check:
        x, meta, y = E.data(11 * 10000 + 10, 4, 0, 40)
        fm = E.load_flow(11)
        pm = E.load_preroute(11)
        b, L, K = 4, 4, 2
        B = 40 // L
        true = E.serial_traj(fm, x, meta)
        tb = [true[i * L].reshape(b, E.N) for i in range(B + 1)]
        for iname, p0 in [('zero', torch.zeros(b, B, 4, M)), ('preroute', preroute_p0(pm, x, meta, L))]:
            ref = E.run_solver_chain(fm, x, meta, y, [t.reshape(b, E.N) for t in p0.unbind(1)],
                                     B, L, rmax=None)
            print(f'--- {iname}: ref acc {ref[K]["acc"]:.4f} Eh {ref[K]["Eh"]:.3e}')
            for tag, rm in [('r64', 64), ('r16', 16), ('r4', 4)]:
                bounds, _ = structured_solve(fm, x, meta, L, K, rm, p0)
                eh = E.eh([t.reshape(b, 4, M) for t in bounds.unbind(1)],
                          [t.reshape(b, 4, M) for t in tb])
                print(f'  {tag}: acc {E.acc_from_state(fm, bounds[:, -1], y):.4f} Eh {eh:.3e}')
        return

    rows = []
    t0 = time.perf_counter()
    for steps in args.steps:
        for L in args.ls:
            if steps % L:
                continue
            for K in args.ks:
                for batch in args.batch:
                    r = bench_case(args, args.seed, args.task, steps, batch, L, K, args.rmax, dev)
                    rows.append(r)
                    def fmt(name):
                        e = r[name]
                        if 'error' in e:
                            return f"{name} ERR"
                        if name == 'serial':
                            return f"serial {e['latency_s']*1e3:.1f}ms/{e.get('acc256', -1):.3f}"
                        return (f"{name} {e['latency_s']*1e3:.1f}ms/"
                                f"{e.get('acc256', float('nan')):.3f}")
                    print(f"T={steps} L={L} K={K} b={batch}: " + " | ".join(fmt(n) for n in
                          ['serial', 'gs_jvp', 'structured', 'structured_seq']), flush=True)
    (ROOT / args.out).write_text(json.dumps({'rows': rows, 'device': str(dev),
                                             'seconds': time.perf_counter() - t0}, indent=2))
    print('wrote', args.out, len(rows), 'rows in', round(time.perf_counter() - t0, 1), 's')


if __name__ == '__main__':
    main()

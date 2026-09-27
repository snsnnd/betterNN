"""第二十轮指标：解析 J_t、ρ_eff、γ_H、state lifetime、可达性（正式任务 + probe）。

J_t = S_t + C_t 的解析构造与 v19 相同，已与 autograd 对齐（v19 Phase 0B）。
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.func import jacfwd, vmap

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import experiment as E

N, M = E.N, E.M
torch.use_deterministic_algorithms(True)


def readout(m, h):
    from torch.nn import functional as F
    return torch.cat([F.linear(h[:, 2], m.headX.weight, m.headX.bias),
                      F.linear(h[:, 3], m.headY.weight, m.headY.bias)], 1)


def step_map(m, h, xt, mt):
    w = (m.W * m.mask).reshape(4, M, 4, M)
    a = .2 * m.hold(mt).sigmoid()
    gate = m.route(torch.cat([mt, h.sum(-1) / M], 1)).sigmoid().reshape(-1, 4, 4)
    rec = (torch.einsum('bsi,sidj->bsdj', h, w) * gate[:, :, :, None]).sum(1)
    cand = torch.tanh(rec + (xt @ m.B).reshape(-1, 4, M))
    h_new = (1 - a[:, :, None]) * h + a[:, :, None] * cand
    return h_new, cand, gate, a


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
        return vmap(jacfwd(f, argnums=1), in_dims=(0, 0))(mt, pooled)


def step_operator(m, h, xt, mt):
    """J = S + C（rank(C)<=4）；返回 J [b,N,N]。"""
    b = len(h)
    w = (m.W * m.mask).reshape(4, M, 4, M)
    _, cand, gate, a = step_map(m, h, xt, mt)
    A = (gate[:, :, None, :, None] * w[None]).permute(0, 3, 4, 1, 2).reshape(b, N, N)
    a_rep = a.repeat_interleave(M, dim=1)
    da = a_rep * (1 - cand.reshape(b, N) ** 2)
    S = (1 - a_rep)[:, :, None] * torch.eye(N) + da[:, :, None] * A
    pooled = h.sum(-1) / M
    G = route_jac(m, mt, pooled)
    Grows = G.reshape(b, 4, 4, 4).permute(0, 2, 1, 3).reshape(b, 16, 4)
    R = torch.einsum('psi,sidj->psdj', h, w)
    X = R.permute(0, 2, 3, 1)
    M1 = torch.zeros(b, N, 16)
    for d in range(4):
        M1[:, d * M:(d + 1) * M, d * 4:(d + 1) * 4] = X[:, d]
    Gp = Grows.repeat_interleave(M, dim=2) / M
    C = da[:, :, None] * (M1 @ Gp)
    return S + C


def jacobians(m, x, meta, samples=2, ts=(4, 8, 12, 16)):
    """取若干起始时刻的 J_t 及路径上的乘积，用于 ρ_eff / γ_H。"""
    with torch.no_grad():
        traj = serial_traj(m, x, meta)
    out = {'t': [], 'rho': [], 'sigma': [], 'gamma4': [], 'gamma8': []}
    for t in ts:
        if t >= x.shape[1]:
            continue
        Js = []
        for k in range(8):
            tt = t + k
            if tt >= x.shape[1]:
                Js.append(None)
                continue
            J = step_operator(m, traj[tt][:samples], x[:samples, tt], meta[:samples, tt])
            Js.append(J)
        rho = [float(torch.linalg.eigvals(Js[0][i]).abs().max()) for i in range(samples)]
        sig = [float(torch.linalg.svdvals(Js[0][i])[0]) for i in range(samples)]
        g4 = [_gamma(Js, i, 4) for i in range(samples)]
        g8 = [_gamma(Js, i, 8) for i in range(samples)]
        out['t'].append(t)
        out['rho'].append(float(np.mean(rho)))
        out['sigma'].append(float(np.mean(sig)))
        out['gamma4'].append(float(np.nanmean(g4)))
        out['gamma8'].append(float(np.nanmean(g8)))
    return out


def _gamma(Js, i, H):
    prod = None
    for k in range(H):
        if Js[k] is None:
            return float('nan')
        prod = Js[k][i] if prod is None else Js[k][i] @ prod
    return math.log(float(torch.linalg.svdvals(prod)[0]) + 1e-12) / H


@torch.no_grad()
def lifetime_metrics(m, x, meta, t_star=8, horizon=24, eps=1e-3, samples=16):
    """参考轨迹用同一 forward；扰动轨迹用相同输入，比较状态差。"""
    x, meta = x[:samples], meta[:samples]
    b = len(x)
    with torch.no_grad():
        traj = serial_traj(m, x, meta)
    h0 = traj[t_star]
    delta0 = torch.randn(b, 4, M) * eps
    h_pert = h0 + delta0
    ref = traj[t_star + 1:]
    ratios = []
    for k in range(min(horizon, len(ref))):
        t = t_star + k
        h_pert, _, _, _ = step_map(m, h_pert, x[:, t], meta[:, t])
        ratios.append(float((h_pert - ref[k]).norm(dim=(1, 2)).mean()))
    ratios = np.array(ratios) / (float(delta0.norm(dim=(1, 2)).mean()) + 1e-12)
    tau = next((k + 1 for k in range(len(ratios)) if ratios[k] <= .5), None)
    ratio_end = float(ratios[-1]) if len(ratios) else float('nan')
    decay = -math.log(max(ratio_end, 1e-12)) / max(len(ratios), 1)
    return {'tau_half': tau, 'G_max': float(ratios.max()), 'ratio_end': ratio_end,
            'decay_rate': decay, 'ratios': [float(r) for r in ratios[:12]]}


@torch.no_grad()
def reachability_task(m, x, meta):
    """正式任务轨迹的边界状态 PCA 有效秩 r_PR = (tr C)^2 / tr(C^2)。"""
    ff = serial_traj(m, x, meta)[-1]
    X = ff.reshape(len(ff), -1)
    X = X - X.mean(0, keepdim=True)
    G = X @ X.t()                                  # [n,n]
    trC = float(torch.trace(G)) / len(X)
    trC2 = float((G / len(X)).pow(2).sum())
    return {'r_eff': (trC ** 2) / (trC2 + 1e-12), 'trC': trC, 'var_explained': trC2}


def reachability_probe(m, steps=20, sigma=0.1, samples=4, task=0, seed=11):
    """非训练 probe：多点小随机输入 + 固定 meta，测 ∂h_T/∂x 的谱。"""
    _, meta, _ = E.data(seed * 10000 + task * 100 + 10, samples, task, steps)
    meta = meta[:samples]
    gen = torch.Generator().manual_seed(seed * 31 + task)
    x0 = torch.randn(samples, steps, 2, generator=gen) * sigma

    def f(xf):
        xx = xf.reshape(samples, steps, 2)

        def one(xs):
            h = torch.zeros(4, M)
            for t in range(steps):
                h = _step_single(m, h, xs[t], meta[0, t])
            return h.reshape(-1)
        return vmap(one)(xx)

    J = jacfwd(f)(x0.reshape(-1))                  # [samples,N,samples*steps*2]
    svs = []
    for i in range(samples):
        Ji = J[i][:, i * steps * 2:(i + 1) * steps * 2]
        svs.append(torch.linalg.svdvals(Ji))
    sv = torch.stack(svs).mean(0)
    e = sv ** 2
    r_eff = float((e.sum() ** 2) / (e.pow(2).sum() + 1e-12))
    with torch.no_grad():
        hT = f(x0.reshape(-1))
    gain = float(hT.norm(dim=1).mean() / (x0.norm(dim=(1, 2)).mean() + 1e-12))
    return {'r_eff': r_eff, 'sigma_max': float(sv[0]), 'sigma_sum': float(sv.sum()),
            'gain': gain}


def _step_single(m, h, xt, mt):
    w = (m.W * m.mask).reshape(4, M, 4, M)
    a = .2 * m.hold(mt).sigmoid()
    gate = m.route(torch.cat([mt, h.sum(-1) / M], 0)).sigmoid().reshape(4, 4)
    rec = (torch.einsum('si,sidj->sdj', h, w) * gate[:, :, None]).sum(0)
    cand = torch.tanh(rec + (xt @ m.B).reshape(4, M))
    return (1 - a[:, None]) * h + a[:, None] * cand


def grad_conflict(m, seed, task_new, task_old, batch=128):
    """两个任务的截断梯度在 W/Route/Hold 上的 cosine（v13 口径）。"""
    from torch.nn import functional as F
    groups = {'W': ('W',), 'route': ('route',), 'hold': ('hold',)}

    def grads(task):
        x, meta, y = E.data(seed * 10000 + task * 100, batch, task)
        m.train()
        m.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(m(x, meta), y)
        loss.backward()
        out = {}
        for g, prefixes in groups.items():
            vec = [p.grad.reshape(-1) for n, p in m.named_parameters()
                   if p.grad is not None and n.split('.')[0] in prefixes]
            out[g] = torch.cat(vec)
        m.zero_grad(set_to_none=True)
        return out

    ga, gb = grads(task_new), grads(task_old)
    out = {}
    for g in groups:
        out[g] = float(torch.nn.functional.cosine_similarity(ga[g], gb[g], dim=0))
    return out


def load_ckpt(path, device='cpu'):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = E.build(ck['seed'], ck.get('sw', .9))
    spec = E.make_B(m, ck['seed'], ck['topo'], ck.get('alpha'))
    m.load_state_dict(ck['state'])
    m.eval()
    return m, ck, spec


def config_metrics(path, task=0, n=256):
    m, ck, _ = load_ckpt(path)
    x, meta, y = E.data(ck['seed'] * 10000 + task * 100 + 10, n, task)
    with torch.no_grad():
        dyn = jacobians(m, x, meta, samples=2)
        life = lifetime_metrics(m, x, meta, samples=16)
        reach_t = reachability_task(m, x, meta)
        probe = reachability_probe(m, seed=ck['seed'], task=task)
    return {'ckpt': str(path), 'topo': ck['topo'], 'alpha': ck.get('alpha'), 'sw': ck.get('sw', .9),
            'seed': ck['seed'], 'task': task,
            'rho_eff': float(np.nanmean(dyn['rho'])), 'sigma_eff': float(np.nanmean(dyn['sigma'])),
            'gamma4': float(np.nanmean(dyn['gamma4'])), 'gamma8': float(np.nanmean(dyn['gamma8'])),
            'tau_half': life['tau_half'], 'G_max': life['G_max'],
            'decay_rate': life['decay_rate'], 'ratio_end': life['ratio_end'],
            'r_PR': reach_t['r_eff'], 'probe_r_eff': probe['r_eff'],
            'probe_sigma_max': probe['sigma_max'], 'probe_gain': probe['gain'],
            'dynamics': dyn}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('--task', type=int, default=0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    res = config_metrics(args.ckpt, args.task)
    text = json.dumps(res, indent=2)
    if args.out:
        Path(args.out).write_text(text)
    print(text)

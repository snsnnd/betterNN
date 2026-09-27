"""第二十一轮：Adaptive Input Decoupling。

- learnable B 只在 role0∪role1（128 维）可写，role2/3 恒 0；
- 训练 B_raw，forward/惩罚/指标全部用 B_eff = β·B_raw/‖B_raw‖；
- penalty：orth（方向 cos²）与 overlap（归一化幅度重叠 L1）；
- 记录每 5 epoch 的 O_B 轨迹与每 stage 的 O_h / cos_* / B drift。
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
WRITE = 2 * M                             # role0 ∪ role1
BETA = .7 * math.sqrt(M)                  # 5.6
SEEDS = [11, 22, 33, 44, 55]
EPOCHS = 65
LR = .003
BATCH = 128
THRESHOLD = .9
BUFFER_PER_TASK = 32
ORDERS = {'o0': [0, 1, 2, 3], 'o1': [3, 2, 1, 0], 'o2': [1, 3, 0, 2]}
RATIOS = {0: '0', .125: '12.5'}
ARMS = ['fixed-overlap', 'fixed-disjoint', 'learnable', 'learnable+orth0.1', 'learnable+overlap0.1',
        'learnable+orth1', 'learnable+overlap1', 'learnable-random']


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
    """Flow-v2 + 可训练 B（仅 role0∪role1）。"""

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
        self.register_buffer('B', b)                      # 固定臂的写入（会被 make_fixed_B 覆盖）
        if learnable:
            raw = b[:, :WRITE].clone() if init_raw is None else init_raw.clone()
            self.B_raw = nn.Parameter(raw)
        else:
            self.B_raw = None
        self.detach_window = True                          # 'full' 模式关闭窗口截断
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
            if self.training and self.detach_window and t and t % 5 == 0:
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
    """V20 shared-pool 构造：两通道写 role0∪role1，每通道 8+8，α 为共享比例。"""
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


def arm_spec(arm):
    """-> (learnable, lam_orth, lam_overlap, init_kind)"""
    if arm == 'fixed-overlap':
        return False, 0, 0, 'overlap'
    if arm == 'fixed-disjoint':
        return False, 0, 0, 'disjoint'
    if arm == 'learnable':
        return True, 0, 0, 'overlap'
    if arm == 'learnable-random':
        return True, 0, 0, 'random'
    if arm.startswith('learnable+orth'):
        return True, float(arm.replace('learnable+orth', '')), 0, 'overlap'
    if arm.startswith('learnable+overlap'):
        return True, 0, float(arm.replace('learnable+overlap', '')), 'overlap'
    raise ValueError(arm)


def make_model(seed, arm, sw=.9):
    learnable, lo, lp, init_kind = arm_spec(arm)
    m = build(seed, sw, learnable)
    if not learnable:
        alpha = 1.0 if init_kind == 'overlap' else 0.0
        with torch.no_grad():
            m.B.copy_(_overlap_B(seed, alpha))
    else:
        if init_kind == 'random':
            g = torch.Generator().manual_seed(seed * 7919 + 17)
            raw = torch.randn(2, WRITE, generator=g)
        else:
            raw = _overlap_B(seed, 1.0)[:, :WRITE].clone()
        with torch.no_grad():
            m.B_raw.copy_(raw)
    return m


def b_metrics(B):
    b = B[:, :WRITE]
    a, c = b[0], b[1]
    na, nc = a.norm() + 1e-12, c.norm() + 1e-12
    o_l1 = float((a * c).abs().sum() / (na * nc))
    cos = float((a @ c) / (na * nc))
    ta = set(torch.topk(a.abs(), 16).indices.tolist())
    tc = set(torch.topk(c.abs(), 16).indices.tolist())
    jac = len(ta & tc) / max(len(ta | tc), 1)
    return {'O_B': o_l1, 'cos_B': cos, 'jaccard16': jac}


def penalties(m, lam_orth, lam_overlap):
    if lam_orth == 0 and lam_overlap == 0:
        return torch.zeros((), device=m.W.device)
    b = m.eff_B()[:, :WRITE]
    a, c = b[0], b[1]
    na, nc = a.norm() + 1e-12, c.norm() + 1e-12
    p = torch.zeros((), device=m.W.device)
    if lam_orth:
        p = p + lam_orth * (a @ c).pow(2) / (na.pow(2) * nc.pow(2))
    if lam_overlap:
        p = p + lam_overlap * (a * c).abs().sum() / (na * nc)
    return p


@torch.no_grad()
def state_at(m, x, meta, t):
    B = m.eff_B()
    h = x.new_zeros(len(x), 4, M)
    w = (m.W * m.mask).reshape(4, M, 4, M)
    for s in range(t):
        ctl = meta[:, s]
        a = .2 * m.hold(ctl).sigmoid()
        gate = m.route(torch.cat([ctl, h.sum(-1) / M], 1)).sigmoid().reshape(-1, 4, 4)
        rec = (torch.einsum('bsi,sidj->bsdj', h, w) * gate[:, :, :, None]).sum(1)
        cand = torch.tanh(rec + (x[:, s] @ B).reshape(-1, 4, M))
        h = (1 - a[:, :, None]) * h + a[:, :, None] * cand
    return h


@torch.no_grad()
def state_overlap(m, seed, task=0, t=8, r=8, samples=64):
    x, meta, _ = data(seed * 10000 + task * 100 + 10, samples, task)
    xA = x.clone()
    xA[:, :, 1] = 0
    xB = x.clone()
    xB[:, :, 0] = 0
    x0 = torch.zeros_like(x)
    dA = (state_at(m, xA, meta, t) - state_at(m, x0, meta, t)).reshape(samples, -1)
    dB = (state_at(m, xB, meta, t) - state_at(m, x0, meta, t)).reshape(samples, -1)
    UA = torch.linalg.svd(dA, full_matrices=False)[2][:r]
    UB = torch.linalg.svd(dB, full_matrices=False)[2][:r]
    return float((UA @ UB.t()).pow(2).sum() / r)


def grad_conflict(m, seed, task_new, task_old, batch=128):
    groups = {'W': ('W',), 'route': ('route',), 'hold': ('hold',), 'B': ('B_raw',)}

    def grads(task):
        x, meta, y = data(seed * 10000 + task * 100, batch, task)
        m.train()
        m.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(m(x, meta), y).backward()
        out = {}
        for g, prefixes in groups.items():
            if g == 'B':
                continue
            vec = [p.grad.reshape(-1) for n, p in m.named_parameters()
                   if p.grad is not None and n.split('.')[0] in prefixes]
            out[g] = torch.cat(vec) if vec else None
        m.zero_grad(set_to_none=True)
        if m.B_raw is not None and 'B' in groups:
            m.eval()
            loss = F.binary_cross_entropy_with_logits(m(x, meta), y)
            out['B'] = torch.autograd.grad(loss, m.B_raw)[0].reshape(-1)
            m.train()
        else:
            out['B'] = None
        return out

    ga, gb = grads(task_new), grads(task_old)
    out = {}
    for g in groups:
        if ga[g] is None or gb[g] is None:
            out[g] = None
        else:
            out[g] = float(F.cosine_similarity(ga[g], gb[g], dim=0))
    return out


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


def _step_loss(m, xb, mb, yb, lam_orth, lam_overlap):
    return (F.binary_cross_entropy_with_logits(m(xb, mb), yb)
            + penalties(m, lam_orth, lam_overlap))


def train_stage(m, seed, task, epochs, lr, replay_n=0, buffer=None, lam_orth=0, lam_overlap=0,
                trace_every=5, b_grad='hybrid'):
    tr = data(seed * 10000 + task * 100, 512, task)
    val = data(seed * 10000 + task * 100 + 1, 256, task)
    core = [p for n, p in m.named_parameters() if n != 'B_raw']
    opt_core = torch.optim.Adam(core, lr=lr)
    opt_b = torch.optim.Adam([m.B_raw], lr=lr) if m.B_raw is not None else None
    gen = torch.Generator().manual_seed(seed + task * 100 + 5000)
    gen_rep = torch.Generator().manual_seed(seed + task * 100 + 9000)
    cur_n = BATCH - replay_n
    has_replay = replay_n > 0 and buffer is not None and len(buffer[0]) > 0
    hh = []
    trace = []
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
            if m.B_raw is not None and b_grad == 'hybrid':
                # 同一参数快照上分别求 core（截断）与 B（全 BPTT）梯度，再各自更新
                opt_core.zero_grad(set_to_none=True)
                opt_b.zero_grad(set_to_none=True)
                loss = _step_loss(m, xb, mb, yb, lam_orth, lam_overlap)
                loss.backward()
                core_grads = {n: p.grad.detach().clone() for n, p in m.named_parameters()
                              if n != 'B_raw' and p.grad is not None}
                for p in m.parameters():
                    p.grad = None
                m.eval()
                loss_full = _step_loss(m, xb, mb, yb, lam_orth, lam_overlap)
                bgrad = torch.autograd.grad(loss_full, m.B_raw)[0]
                m.train()
                for n, p in m.named_parameters():
                    if n != 'B_raw' and n in core_grads:
                        p.grad = core_grads[n]
                m.B_raw.grad = bgrad
                nn.utils.clip_grad_norm_(core, 1.)
                nn.utils.clip_grad_norm_([m.B_raw], 1.)
                opt_core.step()
                opt_b.step()
            elif b_grad == 'full':
                opt_core.zero_grad(set_to_none=True)
                if opt_b is not None:
                    opt_b.zero_grad(set_to_none=True)
                m.detach_window = False
                loss = _step_loss(m, xb, mb, yb, lam_orth, lam_overlap)
                loss.backward()
                m.detach_window = True
                nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.)
                opt_core.step()
                if opt_b is not None:
                    opt_b.step()
            else:
                opt_core.zero_grad(set_to_none=True)
                if opt_b is not None:
                    opt_b.zero_grad(set_to_none=True)
                loss = _step_loss(m, xb, mb, yb, lam_orth, lam_overlap)
                loss.backward()
                nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.)
                opt_core.step()
                if opt_b is not None:
                    opt_b.step()
            loss_sum += float(loss.detach()) / len(order[:4 * cur_n].split(cur_n))
        hh.append({'loss': loss_sum, 'validation': evaluate(m, val)})
        if m.B_raw is not None and ((epoch + 1) % trace_every == 0 or epoch == 0):
            trace.append({'epoch': epoch + 1, **b_metrics(m.eff_B())})
    return hh, trace


def metrics(matrix, order):
    n = len(order)
    acq = float(np.mean([matrix[i][order[i]] for i in range(n)]))
    final = float(np.mean(matrix[-1]))
    forget = float(np.mean([max(matrix[s][t] for s in range(i, n)) - matrix[-1][t]
                            for i, t in enumerate(order[:-1])]))
    bwt = float(np.mean([matrix[-1][t] - matrix[i][t] for i, t in enumerate(order[:-1])]))
    return acq, final, forget, bwt


def run_cl(arm, order_key, ratio, seed, out, epochs, sw=.9, b_grad='hybrid'):
    label = RATIOS[ratio]
    name = f'{arm}_{order_key}_r{label}_{seed}'
    p = out / (name + '.json')
    if p.exists():
        return
    order = ORDERS[order_key]
    replay_n = int(round(BATCH * ratio))
    learnable, lam_orth, lam_overlap, _ = arm_spec(arm)
    m = make_model(seed, arm, sw)
    B0 = m.eff_B().clone()
    b0 = b_metrics(B0)
    buffer = ([], [], [])
    matrix, history, epochs_to_90, traces = [], {}, {}, {}
    stage_metrics, conflicts = [], []
    start = time.perf_counter()
    prev_B = B0.clone()
    for i, task in enumerate(order):
        rn = replay_n if i > 0 else 0
        hh, tr = train_stage(m, seed, task, epochs, LR, rn, buffer, lam_orth, lam_overlap, b_grad=b_grad)
        history[str(task)] = hh
        traces[str(task)] = tr
        epochs_to_90[str(task)] = next((j + 1 for j, h in enumerate(hh) if h['validation'] >= THRESHOLD), None)
        matrix.append(scores(m, seed))
        Bcur = m.eff_B().clone()
        sm = {**b_metrics(Bcur),
              'D_B': float((Bcur - B0).norm() / (B0.norm() + 1e-12)),
              'B_drift_prev': float((Bcur - prev_B).norm() / (prev_B.norm() + 1e-12)),
              'B_raw_norm': float(m.B_raw.norm()) if m.B_raw is not None else 0.0,
              'O_h_t8': state_overlap(m, seed, t=8),
              'O_h_t2': state_overlap(m, seed, t=2), 'O_h_t4': state_overlap(m, seed, t=4),
              'O_h_t20': state_overlap(m, seed, t=20)}
        if i < len(order) - 1:
            sm['conflict'] = grad_conflict(m, seed, order[i + 1], order[i])
        stage_metrics.append(sm)
        prev_B = Bcur
        torch.save({'state': m.state_dict(), 'order': order_key, 'ratio': ratio, 'seed': seed,
                    'stage': i, 'task': task, 'arm': arm, 'sw': sw}, out / f'{name}_stage{i}.pt')
        if ratio > 0:
            trd = data(seed * 10000 + task * 100, 512, task)
            for k in range(3):
                buffer[k].append(trd[k][:BUFFER_PER_TASK])
        print(name, 'stage', i, [round(v, 3) for v in matrix[-1]], 'O_B', round(sm['O_B'], 4),
              flush=True)
    acq, final, forget, bwt = metrics(matrix, order)
    row = {'name': name, 'arm': arm, 'learnable': learnable, 'lam_orth': lam_orth, 'lam_overlap': lam_overlap,
           'order': order_key, 'order_seq': order, 'ratio': ratio, 'replay_n': replay_n, 'seed': seed,
           'B0': b0, 'B_final': b_metrics(m.eff_B()), 'b_trace': traces, 'stage_metrics': stage_metrics,
           'matrix': matrix, 'history': history, 'epochs_to_90': epochs_to_90,
           'acquisition': acq, 'final_mean': final, 'forgetting': forget, 'bwt': bwt,
           'params': sum(p.numel() for p in m.parameters() if p.requires_grad),
           'seconds': time.perf_counter() - start, 'epochs': epochs, 'lr': LR}
    p.write_text(json.dumps(row, indent=2))


def run_single(arm, task, seed, out, epochs, sw=.9, b_grad='hybrid'):
    name = f'single_{arm}_t{task}_{seed}'
    p = out / (name + '.json')
    if p.exists():
        return
    learnable, lam_orth, lam_overlap, _ = arm_spec(arm)
    m = make_model(seed, arm, sw)
    start = time.perf_counter()
    hh, tr = train_stage(m, seed, task, epochs, LR, lam_orth=lam_orth, lam_overlap=lam_overlap, b_grad=b_grad)
    final = evaluate(m, data(seed * 10000 + task * 100 + 10, 1024, task))
    torch.save({'state': m.state_dict(), 'seed': seed, 'task': task, 'arm': arm, 'sw': sw},
               out / f'{name}.pt')
    row = {'name': name, 'arm': arm, 'learnable': learnable, 'lam_orth': lam_orth, 'lam_overlap': lam_overlap,
           'task': task, 'seed': seed, 'final': final, 'B_final': b_metrics(m.eff_B()),
           'b_trace': tr, 'history': hh, 'seconds': time.perf_counter() - start, 'epochs': epochs, 'lr': LR}
    p.write_text(json.dumps(row, indent=2))


def _job(args):
    kind, kw = args
    if kind == 'cl':
        run_cl(**kw)
    else:
        run_single(**kw)


CORE_PREFIX = ('W', 'route', 'hold', 'headX', 'headY')


def credit_check(seeds):
    """Phase 0：trunc 无梯度 / hybrid 有梯度 / core 梯度与 V20 截断一致。"""
    out = []
    for seed in seeds:
        ml = make_model(seed, 'learnable')
        mf = make_model(seed, 'fixed-overlap')
        beq = float((ml.eff_B() - mf.eff_B()).abs().max())
        x, meta, y = data(seed * 10000, 128, 0)

        mf.train()
        mf.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(mf(x, meta), y).backward()
        core_v20 = {n: p.grad.detach().clone() for n, p in mf.named_parameters()
                    if n.split('.')[0] in CORE_PREFIX}

        ml.train()
        ml.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(ml(x, meta), y).backward()
        core_hy = {n: p.grad.detach().clone() for n, p in ml.named_parameters()
                   if n.split('.')[0] in CORE_PREFIX}
        b_trunc = float(ml.B_raw.grad.norm())

        ml.zero_grad(set_to_none=True)
        ml.eval()
        loss = F.binary_cross_entropy_with_logits(ml(x, meta), y)
        b_full = float(torch.autograd.grad(loss, ml.B_raw)[0].norm())

        maxdiff = max(float((core_v20[n] - core_hy[n]).abs().max()) for n in core_v20)
        out.append({'seed': seed, 'B_eff_maxdiff': beq, 'core_grad_maxdiff': maxdiff,
                    'gradB_trunc': b_trunc, 'gradB_hybrid': b_full})
        print(f"seed {seed}: B_eff diff {beq:.2e}, core grad diff {maxdiff:.2e}, "
              f"|gradB| trunc {b_trunc:.3e}, hybrid {b_full:.3e}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='+', default=ARMS[:5])
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--orders', nargs='+', default=list(ORDERS))
    ap.add_argument('--ratios', nargs='+', type=float, default=list(RATIOS))
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--out', default='results/A')
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--no-single', action='store_true')
    ap.add_argument('--b-grad', default='hybrid', choices=['hybrid', 'full', 'trunc'])
    ap.add_argument('--credit-check', action='store_true')
    args = ap.parse_args()
    if args.credit_check:
        res = credit_check(args.seeds)
        (ROOT / 'results/credit_check.json').parent.mkdir(parents=True, exist_ok=True)
        (ROOT / 'results/credit_check.json').write_text(json.dumps(res, indent=2))
        return
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    ratios = [r for r in RATIOS if r in args.ratios]
    print('arms', args.arms, 'ratios', ratios, 'epochs', args.epochs, 'b_grad', args.b_grad, flush=True)
    jobs = []
    for arm in args.arms:
        for seed in args.seeds:
            for order_key in args.orders:
                for ratio in ratios:
                    jobs.append(('cl', dict(arm=arm, order_key=order_key, ratio=ratio, seed=seed,
                                            out=out, epochs=args.epochs, b_grad=args.b_grad)))
            if not args.no_single:
                for task in range(4):
                    jobs.append(('single', dict(arm=arm, task=task, seed=seed, out=out, epochs=args.epochs, b_grad=args.b_grad)))
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

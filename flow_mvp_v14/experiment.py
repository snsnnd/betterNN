"""第十四轮：Write 因果归因。5 种写入模式 × r∈{0,12.5%}。"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).parent
torch.set_num_threads(int(os.environ.get('FLOW_THREADS', '1')))
torch.use_deterministic_algorithms(True)
N = 256
SEEDS = [11, 22, 33, 44, 55]
EPOCHS = 65
LR = .003
BATCH = 128
THRESHOLD = .9
BUFFER_PER_TASK = 32
AUX_LAMBDA = 0.5
AUX_BOUNDARY = 4
ORDER = [0, 1, 2, 3]
WRITE_MODES = ['frozen', 'const1', 'const05', 'full_bptt', 'aux']
RATIOS = {0: '0', .125: '12.5'}
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


class FlowNet(nn.Module):
    def __init__(self, write_mode):
        super().__init__()
        self.write_mode = write_mode
        self.window = None if write_mode == 'full_bptt' else 5
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
        self.write = nn.Sequential(nn.Linear(7, 8), nn.Tanh(), nn.Linear(8, 2))
        self.hold = nn.Linear(7, 4)
        nn.init.zeros_(self.hold.weight)
        nn.init.zeros_(self.hold.bias)
        self.route = nn.Sequential(nn.Linear(11, 16), nn.Tanh(), nn.Linear(16, 16))
        self.headX = nn.Linear(self.m, 1)
        self.headY = nn.Linear(self.m, 1)
        with torch.no_grad():
            self.headX.weight.mul_(5)
            self.headY.weight.mul_(5)

    def forward(self, x, meta, return_aux=False):
        h = x.new_zeros(len(x), 4, self.m)
        w = (self.W * self.mask).reshape(4, self.m, 4, self.m)
        aux_out = None
        for t in range(x.shape[1]):
            if self.training and self.window and t and t % self.window == 0:
                h = h.detach()
            ctl = meta[:, t]
            if self.write_mode == 'const1':
                p = torch.ones(len(x), 2, dtype=x.dtype, device=x.device)
            elif self.write_mode == 'const05':
                p = torch.full((len(x), 2), .5, dtype=x.dtype, device=x.device)
            else:
                p = self.write(ctl).sigmoid()
            a = .2 * self.hold(ctl).sigmoid()
            pooled = h.sum(-1) / self.m
            gate = self.route(torch.cat([ctl, pooled], 1)).sigmoid().reshape(-1, 4, 4)
            rec = (torch.einsum('bsi,sidj->bsdj', h, w) * gate[:, :, :, None]).sum(1)
            cand = torch.tanh(rec + ((x[:, t] * p) @ self.B).reshape(-1, 4, self.m))
            h = (1 - a[:, :, None]) * h + a[:, :, None] * cand
            if return_aux and t == AUX_BOUNDARY:
                aux_out = torch.cat([F.linear(h[:, 2], self.headX.weight, self.headX.bias),
                                     F.linear(h[:, 3], self.headY.weight, self.headY.bias)], 1)
        out = torch.cat([F.linear(h[:, 2], self.headX.weight, self.headX.bias),
                         F.linear(h[:, 3], self.headY.weight, self.headY.bias)], 1)
        return (out, aux_out) if return_aux else out


def build(seed, write_mode):
    torch.manual_seed(seed)
    return FlowNet(write_mode).to(DEVICE)


@torch.no_grad()
def evaluate(m, d):
    m.eval()
    correct = 0
    for ids in torch.arange(len(d[0])).split(128):
        x = d[0][ids].to(DEVICE)
        meta = d[1][ids].to(DEVICE)
        y = d[2][ids].to(DEVICE)
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
                xb = torch.cat([tr[0][chunk], bx[rid]]).to(DEVICE)
                mb = torch.cat([tr[1][chunk], bm[rid]]).to(DEVICE)
                yb = torch.cat([tr[2][chunk], by[rid]]).to(DEVICE)
            else:
                xb = tr[0][chunk].to(DEVICE)
                mb = tr[1][chunk].to(DEVICE)
                yb = tr[2][chunk].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            if m.write_mode == 'aux':
                out, aux = m(xb, mb, return_aux=True)
                loss = F.binary_cross_entropy_with_logits(out, yb) + AUX_LAMBDA * F.binary_cross_entropy_with_logits(aux, yb)
            else:
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
    forget = float(np.mean([max(matrix[s][t] for s in range(i, n)) - matrix[-1][t] for i, t in enumerate(order[:-1])]))
    bwt = float(np.mean([matrix[-1][t] - matrix[i][t] for i, t in enumerate(order[:-1])]))
    return acq, final, forget, bwt


def write_params(m):
    return [p.detach().cpu() for p in m.write.parameters()]


def run_stream(write_mode, ratio, seed, out, epochs):
    label = RATIOS[ratio]
    name = f'{write_mode}_r{label}_{seed}'
    p = out / (name + '.json')
    if p.exists():
        return
    replay_n = int(round(BATCH * ratio))
    m = build(seed, write_mode)
    write_init = [w.clone() for w in write_params(m)]
    buffer = ([], [], [])
    matrix = []
    history = {}
    epochs_to_90 = {}
    start = time.perf_counter()
    for i, task in enumerate(ORDER):
        rn = replay_n if i > 0 else 0
        hh = train_stage(m, seed, task, epochs, LR, rn, buffer)
        history[str(task)] = hh
        epochs_to_90[str(task)] = next((j + 1 for j, h in enumerate(hh) if h['validation'] >= THRESHOLD), None)
        matrix.append(scores(m, seed))
        torch.save({'state': m.state_dict(), 'write_mode': write_mode, 'ratio': ratio, 'seed': seed, 'stage': i, 'task': task},
                   out / f'{name}_stage{i}.pt')
        if ratio > 0:
            tr = data(seed * 10000 + task * 100, 512, task)
            for k in range(3):
                buffer[k].append(tr[k][:BUFFER_PER_TASK])
        print(name, 'stage', i, [round(v, 3) for v in matrix[-1]], flush=True)
    write_delta = float(torch.cat([(a - b).reshape(-1) for a, b in zip(write_params(m), write_init)]).norm())
    acq, final, forget, bwt = metrics(matrix, ORDER)
    row = {'name': name, 'write_mode': write_mode, 'ratio': ratio, 'ratio_label': label, 'replay_n': replay_n,
           'seed': seed, 'matrix': matrix, 'history': history, 'epochs_to_90': epochs_to_90,
           'acquisition': acq, 'final_mean': final, 'forgetting': forget, 'bwt': bwt,
           'write_delta': write_delta, 'aux_lambda': AUX_LAMBDA, 'aux_boundary': AUX_BOUNDARY,
           'buffer_per_task': BUFFER_PER_TASK, 'seconds': time.perf_counter() - start,
           'threads': torch.get_num_threads(), 'device': str(DEVICE), 'torch': torch.__version__,
           'epochs': epochs, 'lr': LR}
    p.write_text(json.dumps(row, indent=2))


def pick_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write-modes', nargs='+', choices=WRITE_MODES, default=WRITE_MODES)
    ap.add_argument('--ratios', nargs='+', type=float, default=list(RATIOS))
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--out', default='results')
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()
    global DEVICE
    DEVICE = pick_device(args.device)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    ratios = [r for r in RATIOS if r in args.ratios]
    (out / 'config.json').write_text(json.dumps(
        {'n': N, 'seeds': SEEDS, 'write_modes': WRITE_MODES, 'ratios': RATIOS, 'order': ORDER,
         'epochs': args.epochs, 'lr': LR, 'batch': BATCH, 'buffer_per_task': BUFFER_PER_TASK,
         'aux_lambda': AUX_LAMBDA, 'aux_boundary': AUX_BOUNDARY, 'torch': torch.__version__,
         'device': str(DEVICE)}, indent=2))
    print('device', DEVICE, 'write_modes', args.write_modes, 'ratios', ratios, 'epochs', args.epochs, flush=True)
    for write_mode in args.write_modes:
        for ratio in ratios:
            for seed in args.seeds:
                run_stream(write_mode, ratio, seed, out, args.epochs)


if __name__ == '__main__':
    main()

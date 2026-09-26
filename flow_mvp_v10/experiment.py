"""第十轮：benchmark校准。scratch/adapt单任务训练，CL基线，验证选预算。"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, copy, json, time
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
EPOCHS = 150
LR = .003
BATCH = 128
THRESHOLD = .9
MODES = ['pretrain', 'scratch', 'adapt', 'cl']
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
    # A'B'C'D' benchmark（R=0/R=1 的输出通道）：A'=(A,B)/(B,A)，B'=(A,A)/(B,B)，
    # C'=(A,B)/(A,A)，D'=(A,B)/(B,B)；不存在互为逆映射的任务对。
    idx = torch.tensor([[[0, 1], [1, 0]], [[0, 0], [1, 1]],
                        [[0, 1], [0, 0]], [[0, 1], [1, 1]]])[task, r]
    return x, meta, (ab.gather(1, idx) > 0).float()


class Net(nn.Module):
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

    def forward(self, x, meta):
        h = x.new_zeros(len(x), 4, self.m)
        w = (self.W * self.mask).reshape(4, self.m, 4, self.m)
        for t in range(x.shape[1]):
            if self.training and t and t % 5 == 0:
                h = h.detach()
            ctl = meta[:, t]
            p = self.write(ctl).sigmoid()
            a = .2 * self.hold(ctl).sigmoid()
            pooled = h.sum(-1) / self.m
            gate = self.route(torch.cat([ctl, pooled], 1)).sigmoid().reshape(-1, 4, 4)
            rec = (torch.einsum('bsi,sidj->bsdj', h, w) * gate[:, :, :, None]).sum(1)
            cand = torch.tanh(rec + ((x[:, t] * p) @ self.B).reshape(-1, 4, self.m))
            h = (1 - a[:, :, None]) * h + a[:, :, None] * cand
        return torch.cat([F.linear(h[:, 2], self.headX.weight, self.headX.bias),
                          F.linear(h[:, 3], self.headY.weight, self.headY.bias)], 1)


def build(seed):
    torch.manual_seed(seed)
    return Net()


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


def train(m, seed, task, epochs, lr):
    tr = data(seed * 10000 + task * 100, 512, task)
    val = data(seed * 10000 + task * 100 + 1, 256, task)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=lr)
    gen = torch.Generator().manual_seed(seed + task * 100 + 5000)
    hh = []
    best = (-1., None, None)
    for epoch in range(epochs):
        m.train()
        loss_sum = 0
        for ids in torch.randperm(512, generator=gen).split(BATCH):
            xb = tr[0][ids].to(DEVICE)
            mb = tr[1][ids].to(DEVICE)
            yb = tr[2][ids].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(m(xb, mb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.)
            opt.step()
            loss_sum += float(loss.detach()) / 4
        v = evaluate(m, val)
        hh.append({'loss': loss_sum, 'validation': v})
        if v > best[0]:
            best = (v, copy.deepcopy(m.state_dict()), epoch + 1)
    return hh, best


def save_run(out, sub, name, m, best, hh, seed, task):
    d = out / sub
    d.mkdir(parents=True, exist_ok=True)
    torch.save({'best': best[1], 'final': m.state_dict(), 'seed': seed, 'task': task}, d / (name + '.pt'))
    return {'seed': seed, 'task': task, 'best_val': best[0], 'best_epoch': best[2],
            'history': hh, 'test_final': scores(m, seed)}


def finalize(m, out, sub, name, best, hh, seed, task, lr):
    row = save_run(out, sub, name, m, best, hh, seed, task)
    row['test_best'] = None
    m_best = build(seed).to(DEVICE)
    m_best.load_state_dict(best[1])
    row['test_best'] = scores(m_best, seed)
    row['threads'] = torch.get_num_threads()
    row['device'] = str(DEVICE)
    row['lr'] = lr
    (out / sub / (name + '.json')).write_text(json.dumps(row, indent=2))
    return row


def run_pretrain(seed, out, epochs, lr):
    p = out / 'pretrain' / f'{seed}.json'
    if p.exists():
        return
    m = build(seed).to(DEVICE)
    t0 = time.perf_counter()
    hh, best = train(m, seed, 0, epochs, lr)
    row = finalize(m, out, 'pretrain', f'{seed}', best, hh, seed, 0, lr)
    row['seconds'] = time.perf_counter() - t0
    p.write_text(json.dumps(row, indent=2))
    print('pretrain', seed, 'best', round(row['best_val'], 3), 'testA', round(row['test_final'][0], 3), flush=True)


def run_scratch(seed, task, out, epochs, lr):
    p = out / 'scratch' / f'{task}_{seed}.json'
    if p.exists():
        return
    m = build(seed).to(DEVICE)
    t0 = time.perf_counter()
    hh, best = train(m, seed, task, epochs, lr)
    row = finalize(m, out, 'scratch', f'{task}_{seed}', best, hh, seed, task, lr)
    row['seconds'] = time.perf_counter() - t0
    p.write_text(json.dumps(row, indent=2))
    print('scratch', task, seed, 'best', round(row['best_val'], 3), 'test', round(row['test_final'][task], 3), flush=True)


def run_adapt(seed, task, out, epochs, lr):
    p = out / 'adapt' / f'{task}_{seed}.json'
    if p.exists():
        return
    pre = torch.load(out / 'pretrain' / f'{seed}.pt', map_location='cpu', weights_only=False)['final']
    m = build(seed)
    m.load_state_dict(pre)
    m = m.to(DEVICE)
    t0 = time.perf_counter()
    hh, best = train(m, seed, task, epochs, lr)
    row = finalize(m, out, 'adapt', f'{task}_{seed}', best, hh, seed, task, lr)
    row['seconds'] = time.perf_counter() - t0
    p.write_text(json.dumps(row, indent=2))
    print('adapt', task, seed, 'best', round(row['best_val'], 3), 'test', round(row['test_final'][task], 3), flush=True)


def run_cl(seed, out, epochs, lr_map):
    p = out / 'cl' / f'{seed}.json'
    if p.exists():
        return
    (out / 'cl').mkdir(parents=True, exist_ok=True)
    m = build(seed).to(DEVICE)
    matrix = []
    history = {}
    epochs_to_90 = {}
    t0 = time.perf_counter()
    for task in range(4):
        hh, _ = train(m, seed, task, epochs, lr_map[task])
        history[str(task)] = hh
        epochs_to_90[str(task)] = next((i + 1 for i, h in enumerate(hh) if h['validation'] >= THRESHOLD), None)
        matrix.append(scores(m, seed))
        torch.save({'state': m.state_dict(), 'seed': seed, 'stage': task}, out / 'cl' / f'{seed}_stage{task}.pt')
        print('cl', seed, 'stage', task, [round(v, 3) for v in matrix[-1]], flush=True)
    row = {'seed': seed, 'matrix': matrix, 'history': history, 'epochs_to_90': epochs_to_90,
           'acquisition': float(np.mean([matrix[t][t] for t in range(4)])),
           'final_mean': float(np.mean(matrix[-1])),
           'forgetting': float(np.mean([max(matrix[s][t] for s in range(t, 4)) - matrix[-1][t] for t in range(3)])),
           'bwt': float(np.mean([matrix[-1][t] - matrix[t][t] for t in range(3)])),
           'seconds': time.perf_counter() - t0, 'threads': torch.get_num_threads(),
           'device': str(DEVICE), 'torch': torch.__version__, 'epochs': epochs, 'lr_map': lr_map}
    p.write_text(json.dumps(row, indent=2))


def pick_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modes', nargs='+', choices=MODES, default=MODES)
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--lr', type=float, default=LR)
    ap.add_argument('--lr-map', nargs=4, type=float, default=None,
                    help='按任务A/B/C/D的学习率；省略时所有任务用 --lr')
    ap.add_argument('--out', default='results')
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()
    global DEVICE
    DEVICE = pick_device(args.device)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    lr_map = args.lr_map or [args.lr] * 4
    (out / 'config.json').write_text(json.dumps(
        {'n': N, 'epochs': args.epochs, 'lr_map': lr_map, 'batch': BATCH, 'threshold': THRESHOLD,
         'torch': torch.__version__, 'device': str(DEVICE)}, indent=2))
    print('device', DEVICE, 'torch', torch.__version__, 'epochs', args.epochs, 'lr_map', lr_map, flush=True)
    for seed in args.seeds:
        if 'pretrain' in args.modes:
            run_pretrain(seed, out, args.epochs, lr_map[0])
        for task in range(4):
            if 'scratch' in args.modes:
                run_scratch(seed, task, out, args.epochs, lr_map[task])
            if task > 0 and 'adapt' in args.modes:
                run_adapt(seed, task, out, args.epochs, lr_map[task])
        if 'cl' in args.modes:
            run_cl(seed, out, args.epochs, lr_map)


if __name__ == '__main__':
    main()

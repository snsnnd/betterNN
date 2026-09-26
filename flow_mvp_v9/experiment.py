"""第九轮：遗忘定位。A全参数预训练；B/C/D分别保护W/控制器/读出并加固定预算replay。"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, copy, json, math, time
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
EPOCHS = 30
THRESHOLD = 0.9
BATCH = 128
REPLAY_PER_OLD = 32
REPLAY_BATCH = 32
ARMS = ['all', 'freeze_W', 'freeze_ctrl', 'freeze_readout', 'ctrl_only', 'readout_only', 'replay']
TRAINABLE = {'all': ['W', 'ctrl', 'readout'], 'freeze_W': ['ctrl', 'readout'],
             'freeze_ctrl': ['W', 'readout'], 'freeze_readout': ['W', 'ctrl'],
             'ctrl_only': ['ctrl'], 'readout_only': ['readout'],
             'replay': ['W', 'ctrl', 'readout']}
NAME_TO_GROUP = {'W': 'W', 'write': 'ctrl', 'hold': 'ctrl', 'route': 'ctrl',
                 'headX': 'readout', 'headY': 'readout'}
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
    idx = torch.tensor([[[0, 1], [1, 0]], [[0, 0], [1, 1]], [[1, 1], [0, 0]], [[1, 0], [0, 1]]])[task, r]
    return x, meta, (ab.gather(1, idx) > 0).float()


class Net(nn.Module):
    """与第八轮 dense 相同的结构；selector 保留但不参与本轮训练。"""

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
        self.selector = nn.Parameter(torch.randn(4, 4, self.m) * .01, requires_grad=False)

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


def groups(m):
    out = {'W': [m.W], 'ctrl': [], 'readout': []}
    for n, p in m.named_parameters():
        g = NAME_TO_GROUP.get(n.split('.')[0])
        if g and g != 'W':
            out[g].append(p)
    return out


def build(seed, arm):
    torch.manual_seed(seed)
    m = Net()
    for name, ps in groups(m).items():
        for p in ps:
            p.requires_grad_(name in TRAINABLE[arm])
    return m


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


def train_stage(m, seed, task, epochs, replay=False, buffer=None):
    tr = data(seed * 10000 + task * 100, 512, task)
    val = data(seed * 10000 + task * 100 + 1, 256, task)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=.003)
    gen = torch.Generator().manual_seed(seed + task * 100 + 5000)
    hh = []
    for epoch in range(epochs):
        m.train()
        loss_sum = 0
        order = torch.randperm(512, generator=gen)
        if replay:
            chunks = order[:384].split(96)
        else:
            chunks = order.split(BATCH)
        for chunk in chunks:
            if replay and buffer is not None and len(buffer[0]):
                bx, bm, by = (torch.cat(b) for b in buffer)
                rid = torch.randint(len(bx), (REPLAY_BATCH,), generator=gen)
                xb = torch.cat([tr[0][chunk], bx[rid]]).to(DEVICE)
                mb = torch.cat([tr[1][chunk], bm[rid]]).to(DEVICE)
                yb = torch.cat([tr[2][chunk], by[rid]]).to(DEVICE)
            else:
                xb = tr[0][chunk].to(DEVICE)
                mb = tr[1][chunk].to(DEVICE)
                yb = tr[2][chunk].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(m(xb, mb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.)
            opt.step()
            loss_sum += float(loss.detach()) / len(chunks)
        hh.append({'loss': loss_sum, 'validation': evaluate(m, val)})
    return hh


def group_delta(m, pre):
    delta = {g: 0.0 for g in ['W', 'ctrl', 'readout']}
    for n, p in m.named_parameters():
        g = NAME_TO_GROUP.get(n.split('.')[0])
        if g:
            delta[g] += float((p.detach().cpu() - pre[n]).pow(2).sum())
    return {g: math.sqrt(v) for g, v in delta.items()}


def scores(m, seed):
    return [evaluate(m, data(seed * 10000 + t * 100 + 10, 1024, t)) for t in range(4)]


def run_pretrain(seed, out):
    p = out / 'pretrain' / f'{seed}.json'
    if p.exists():
        return
    (out / 'pretrain').mkdir(exist_ok=True)
    m = build(seed, 'all').to(DEVICE)
    start = time.perf_counter()
    hh = train_stage(m, seed, 0, EPOCHS)
    row = {'seed': seed, 'matrix': [scores(m, seed)], 'history': {'0': hh},
           'epochs_to_90': {'0': next((i + 1 for i, h in enumerate(hh) if h['validation'] >= THRESHOLD), None)},
           'seconds': time.perf_counter() - start}
    torch.save({'state': m.state_dict(), 'seed': seed}, out / 'pretrain' / f'{seed}.pt')
    p.write_text(json.dumps(row, indent=2))
    print('pretrain', seed, [round(v, 3) for v in row['matrix'][0]], flush=True)


def run_arm(seed, arm, out, epochs):
    p = out / f'{arm}_{seed}.json'
    if p.exists():
        return
    pre = json.loads((out / 'pretrain' / f'{seed}.json').read_text())
    pre_state = torch.load(out / 'pretrain' / f'{seed}.pt', map_location='cpu', weights_only=False)['state']
    m = build(seed, arm)
    m.load_state_dict(pre_state)
    m = m.to(DEVICE)
    matrix = [list(pre['matrix'][0])]
    history = dict(pre['history'])
    epochs_to_90 = dict(pre['epochs_to_90'])
    buffer = ([], [], [])
    start = time.perf_counter()
    for task in [1, 2, 3]:
        hh = train_stage(m, seed, task, epochs, replay=(arm == 'replay'), buffer=buffer)
        history[str(task)] = hh
        epochs_to_90[str(task)] = next((i + 1 for i, h in enumerate(hh) if h['validation'] >= THRESHOLD), None)
        matrix.append(scores(m, seed))
        for k, v in {'B': pre_state['B'], 'mask': pre_state['mask']}.items():
            assert torch.equal(v.cpu(), m.state_dict()[k].cpu())
        if arm == 'replay':
            tr = data(seed * 10000 + task * 100, 512, task)
            for i in range(3):
                buffer[i].append(tr[i][:REPLAY_PER_OLD])
        torch.save({'state': m.state_dict(), 'seed': seed, 'arm': arm, 'stage': task},
                   out / f'{arm}_{seed}_stage{task}.pt')
        print(arm, seed, 'stage', task, [round(v, 3) for v in matrix[-1]], flush=True)
    row = {'seed': seed, 'arm': arm, 'matrix': matrix, 'history': history, 'epochs_to_90': epochs_to_90,
           'acquisition_A': matrix[0][0],
           'acquisition_BCD': float(np.mean([matrix[t][t] for t in [1, 2, 3]])),
           'final_mean': float(np.mean(matrix[-1])),
           'forgetting': float(np.mean([max(matrix[s][t] for s in range(t, 4)) - matrix[-1][t] for t in range(3)])),
           'bwt': float(np.mean([matrix[-1][t] - matrix[t][t] for t in range(3)])),
           'group_delta': group_delta(m, pre_state),
           'trainable': sum(q.numel() for q in m.parameters() if q.requires_grad),
           'seconds': time.perf_counter() - start, 'threads': torch.get_num_threads(),
           'device': str(DEVICE), 'torch': torch.__version__}
    p.write_text(json.dumps(row, indent=2))


def run_control(seed, task, out, epochs):
    p = out / f'control_{seed}_{task}.json'
    if p.exists():
        return
    pre_state = torch.load(out / 'pretrain' / f'{seed}.pt', map_location='cpu', weights_only=False)['state']
    m = build(seed, 'all')
    m.load_state_dict(pre_state)
    m = m.to(DEVICE)
    hh = train_stage(m, seed, task, epochs)
    row = {'seed': seed, 'task': task, 'scores': scores(m, seed),
           'epochs_to_90': next((i + 1 for i, h in enumerate(hh) if h['validation'] >= THRESHOLD), None),
           'history': hh}
    torch.save({'state': m.state_dict(), 'seed': seed, 'task': task}, out / f'control_{seed}_{task}.pt')
    p.write_text(json.dumps(row, indent=2))
    print('control', seed, task, round(row['scores'][task], 3), flush=True)


def pick_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--arms', nargs='+', choices=ARMS, default=ARMS)
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--out', default='results')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--skip-controls', action='store_true')
    args = ap.parse_args()
    global DEVICE
    DEVICE = pick_device(args.device)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(
        {'n': N, 'seeds': SEEDS, 'arms': ARMS, 'trainable': TRAINABLE, 'epochs_per_stage': EPOCHS,
         'train': 512, 'val': 256, 'test': 1024, 'batch': BATCH, 'replay': '96 current + 32 old, same 128 budget',
         'replay_per_old': REPLAY_PER_OLD, 'bptt_window': 5, 'lr': .003, 'pretrain': 'A with all params',
         'selection': 'final epoch', 'torch': torch.__version__, 'device': str(DEVICE)}, indent=2))
    print('device', DEVICE, 'torch', torch.__version__, flush=True)
    for seed in args.seeds:
        run_pretrain(seed, out)
        for arm in args.arms:
            run_arm(seed, arm, out, args.epochs)
        if not args.skip_controls:
            for task in [1, 2, 3]:
                run_control(seed, task, out, args.epochs)


if __name__ == '__main__':
    main()

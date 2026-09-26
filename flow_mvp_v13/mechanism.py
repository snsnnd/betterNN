"""第十三轮共用：数据、Flow 模型与评估。与第十二轮 flow 路径逐位一致。"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn
from torch.nn import functional as F

torch.use_deterministic_algorithms(True)
N = 256
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
    return FlowNet().to(DEVICE)


def group_params(m):
    return {'W': [m.W], 'write': list(m.write.parameters()), 'hold': list(m.hold.parameters()),
            'route': list(m.route.parameters()),
            'readout': list(m.headX.parameters()) + list(m.headY.parameters())}


def group_names(m):
    names = {'W': ['W'], 'ctrl': [], 'readout': []}
    for n, _ in m.named_parameters():
        root = n.split('.')[0]
        if root in ('write', 'hold', 'route'):
            names['ctrl'].append(n)
        elif root in ('headX', 'headY'):
            names['readout'].append(n)
    return names


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

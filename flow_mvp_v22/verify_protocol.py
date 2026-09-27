"""V22 协议核验：B 逐位回归、All/V21D 对照、首 batch 梯度等价、clip 生效证明。"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import experiment as E  # noqa: E402

torch.set_num_threads(1)


def matrix_diff(a, b):
    return float(np.max(np.abs(np.array(a['matrix']) - np.array(b['matrix']))))


def reg_check():
    out = {}
    for arm, phase, ref in [('B', 'A', 'learnable'), ('All', 'D', 'learnable')]:
        mine = json.loads((ROOT / f'results/reg/{arm}_o0_r12.5_11.json').read_text())
        v21 = json.loads((ROOT / f'../flow_mvp_v21/results/{phase}/{ref}_o0_r12.5_11.json').read_text())
        out[arm] = {'maxdiff': matrix_diff(mine, v21),
                    'forget_v22': mine['forgetting'] * 100, 'forget_v21': v21['forgetting'] * 100}
    return out


def grad_equivalence():
    """首 batch：autograd.grad(all, full) 与 V21D backward 的梯度应逐位相同。"""
    m = E.make_model(11)
    tr = E.data(11 * 10000, 512, 0)
    gen = torch.Generator().manual_seed(11 + 5000)
    order = torch.randperm(512, generator=gen)
    xb, mb, yb = tr[0][order[:128]], tr[1][order[:128]], tr[2][order[:128]]
    m.eval()
    m.detach_window = False
    loss = F.binary_cross_entropy_with_logits(m(xb, mb), yb)
    ga = torch.autograd.grad(loss, [p for _, p in m.named_parameters()])
    m.train()
    m.zero_grad(set_to_none=True)
    loss2 = F.binary_cross_entropy_with_logits(m(xb, mb), yb)
    loss2.backward()
    diff = max(float((x - p.grad).abs().max()) for x, (_, p) in zip(ga, m.named_parameters()))
    return {'loss_bitwise_equal': float(loss) == float(loss2), 'grad_maxdiff': diff}


def clip_binding(steps=260):
    """跑一个 stage 长度的 V22 All 训练，记录最大 pre-clip 范数（>1 表示 clip 生效）。"""
    m = E.make_model(11)
    tr = E.data(11 * 10000, 512, 0)
    gen = torch.Generator().manual_seed(11 + 5000)
    core = [p for n, p in m.named_parameters() if n != 'B_raw']
    opt_core = torch.optim.Adam(core, lr=E.LR)
    opt_b = torch.optim.Adam([m.B_raw], lr=E.LR)
    named = list(m.named_parameters())
    full = E.full_set_of('All')
    mx = 0.0
    for _ in range(steps // 4):
        m.train()
        order = torch.randperm(512, generator=gen)
        for chunk in order[:512].split(128):
            xb, mb, yb = tr[0][chunk], tr[1][chunk], tr[2][chunk]
            opt_core.zero_grad(set_to_none=True)
            opt_b.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(m(xb, mb), yb).backward()
            for p in m.parameters():
                p.grad = None
            m.eval()
            lf = F.binary_cross_entropy_with_logits(m(xb, mb), yb)
            fg = torch.autograd.grad(lf, [p for n, p in named if E.group_of(n) in full])
            m.train()
            gi = 0
            for n, p in named:
                if E.group_of(n) in full:
                    p.grad = fg[gi]
                    gi += 1
            mx = max(mx, float(torch.nn.utils.clip_grad_norm_(core, 1.)))
            torch.nn.utils.clip_grad_norm_([m.B_raw], 1.)
            opt_core.step()
            opt_b.step()
    return {'max_preclip_core_norm': mx, 'clip_binds': mx > 1.0}


if __name__ == '__main__':
    out = {'reg': reg_check(), 'grad_equivalence': grad_equivalence(), 'clip': clip_binding()}
    print(json.dumps(out, indent=1))

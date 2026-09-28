"""V24B 协议核验：scheduler 初始化等价、w≡1、forward 逐值一致、hybrid 下 w 有梯度。"""
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import experiment as E  # noqa: E402
import scheduler as S  # noqa: E402

torch.set_num_threads(1)


def init_equivalence():
    out = {}
    for kernel in ('single', 'burst5', 'decay-slow'):
        torch.manual_seed(11)
        a = E.Flow(.9, learnable=True, kernel_name=kernel)
        torch.manual_seed(11)
        b = S.FlowS(.9, learnable=True, kernel_name=kernel, scheduler=True)
        base = [p for n, p in a.named_parameters()]
        base_s = [p for n, p in b.named_parameters() if not n.startswith('wctl.')]
        pdiff = max(float((x - y).abs().max()) for x, y in zip(base, base_s))
        x, meta, y = E.data_chain(11 * 10000, 64, 'xor', 80)
        with torch.no_grad():
            a.eval(); b.eval()
            fdiff = float((a(x, meta) - b(x, meta)).abs().max())
            w = b.w_values(x, meta)
        out[kernel] = {'param_maxdiff_vs_Flow': pdiff, 'forward_maxdiff_vs_Flow': fdiff,
                       'w_min': float(w.min()), 'w_max': float(w.max())}
    return out


def grad_flow():
    m = S.build_s(11, .9, learnable=True, kernel_name='burst5', scheduler=True)
    with torch.no_grad():
        m.B_raw.copy_(E.make_fixed_B_raw(11))
    tr = E.data_chain(11 * 10000, 128, 'xor', 80)
    named = list(m.named_parameters())
    full_names = {n for n, _ in named if n == 'B_raw' or n.startswith('wctl.')}
    full_params = [p for n, p in named if n in full_names]
    m.zero_grad(set_to_none=True)
    m.train()
    m.detach_window = True
    F.binary_cross_entropy_with_logits(m(tr[0], tr[1]), tr[2]).backward()
    gt = {n: float(p.grad.norm()) for n, p in named if p.grad is not None}
    for p in m.parameters():
        p.grad = None
    m.eval()
    m.detach_window = False
    lf = F.binary_cross_entropy_with_logits(m(tr[0], tr[1]), tr[2])
    gf = torch.autograd.grad(lf, full_params)
    m.zero_grad(set_to_none=True)
    full_grads = {n: float(g.norm()) for (n, _), g in zip(
        [(n, p) for n, p in named if n in full_names], gf)}
    return {'wctl_grad_trunc': {n: v for n, v in gt.items() if n.startswith('wctl.')},
            'wctl_grad_full': {n: v for n, v in full_grads.items() if n.startswith('wctl.')},
            'B_raw_grad_trunc': gt.get('B_raw', 0.0), 'B_raw_grad_full': full_grads.get('B_raw', 0.0)}


if __name__ == '__main__':
    out = {'init_equivalence': init_equivalence(), 'grad_flow': grad_flow()}
    print(json.dumps(out, indent=1))

"""V24C 协议核验：base 参数等价、初始 α/前向≈single、等能量、hybrid 下 ∇α 非零。"""
import json
import math
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import experiment as E  # noqa: E402
import bank as B  # noqa: E402

torch.set_num_threads(1)


def init_check():
    torch.manual_seed(11)
    ref = E.Flow(.9, learnable=True, kernel_name='single')
    torch.manual_seed(11)
    mb = B.FlowB(.9, learnable=True)
    base_ref = [p for _, p in ref.named_parameters()]
    base_b = [p for n, p in mb.named_parameters() if not n.startswith('wbank.')]
    pdiff = max(float((a - b).abs().max()) for a, b in zip(base_ref, base_b))
    x, meta, y = E.data_chain(11 * 10000, 64, 'xor', 80)
    with torch.no_grad():
        ref.eval(); mb.eval()
        fdiff = float((ref(x, meta) - mb(x, meta)).abs().max())
        al = mb.alpha(x, meta)
        tail = float((mb.mixed_kernel(x, meta) - B.BANK['single']).abs().max())
    return {'base_param_maxdiff': pdiff, 'forward_maxdiff_vs_fixed_single': fdiff,
            'alpha_mean': [round(v, 4) for v in al.mean((0, 1)).tolist()],
            'mixed_kernel_tail_maxdiff': tail}


def energy_check():
    """单脉冲输入下，bank 初始化的总写入 L2 能量应等于 fixed single。"""
    torch.manual_seed(11)
    ref = E.Flow(.9, learnable=True, kernel_name='single')
    torch.manual_seed(11)
    mb = B.FlowB(.9, learnable=True)
    with torch.no_grad():
        E.make_fixed_B(ref, 11, 0.0)
        E.make_fixed_B(mb, 11, 0.0)
        x = torch.zeros(4, 40, 2); x[:, 10, 0] = torch.tensor([1., -2., .5, 1.5])
        meta = torch.zeros(4, 40, 7); meta[:, :, 2] = torch.linspace(0, 1, 40)
        w_ref = (x @ ref.B).reshape(4, 40, 4, E.M)
        w_bank = mb.writes(x, meta)
        e_ref = float(w_ref.pow(2).sum())
        e_bank = float(w_bank.pow(2).sum())
    return {'E_fixed_single': e_ref, 'E_bank_init': e_bank, 'rel_diff': abs(e_bank - e_ref) / e_ref}


def grad_flow():
    m = B.build_bank(11, .9, learnable=True)
    with torch.no_grad():
        m.B_raw.copy_(E.make_fixed_B_raw(11))
    tr = E.data_chain(11 * 10000, 128, 'xor', 80)
    named = list(m.named_parameters())
    full_names = {n for n, _ in named if n == 'B_raw' or n.startswith('wbank.')}
    full_params = [p for n, p in named if n in full_names]
    m.zero_grad(set_to_none=True)
    m.train(); m.detach_window = True
    F.binary_cross_entropy_with_logits(m(tr[0], tr[1]), tr[2]).backward()
    gt = {n: float(p.grad.norm()) for n, p in named if p.grad is not None}
    for p in m.parameters():
        p.grad = None
    m.eval(); m.detach_window = False
    lf = F.binary_cross_entropy_with_logits(m(tr[0], tr[1]), tr[2])
    gf = torch.autograd.grad(lf, full_params)
    m.zero_grad(set_to_none=True)
    full_grads = {n: float(g.norm()) for (n, _), g in zip(
        [(n, p) for n, p in named if n in full_names], gf)}
    return {'wbank_trunc': {n: v for n, v in gt.items() if n.startswith('wbank.')},
            'wbank_full': {n: v for n, v in full_grads.items() if n.startswith('wbank.')},
            'B_raw_trunc': gt.get('B_raw', 0.0), 'B_raw_full': full_grads.get('B_raw', 0.0)}


if __name__ == '__main__':
    print(json.dumps({'init': init_check(), 'energy': energy_check(), 'grad': grad_flow()}, indent=1))

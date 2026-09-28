"""V25 Phase 0 协议核验：base 参数等价、初始化核、等能量、hybrid 下 ∇s/∇wbank 非零。"""
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'flow_mvp_v24'))
import experiment as E  # noqa: E402
import audit as A  # noqa: E402

torch.set_num_threads(1)


def init_check():
    seed = 11
    torch.manual_seed(seed)
    ref = E.Flow(.9, learnable=False, kernel_name='single')
    torch.manual_seed(seed)
    mb = A.BankAudit(.9, learnable=False, init='single')
    torch.manual_seed(seed)
    ms = A.SharedLambda(.9, learnable=False, lam0=0.05)
    with torch.no_grad():
        base_ref = [p for n, p in ref.named_parameters()]
        base_b = [p for n, p in mb.named_parameters() if not n.startswith('wbank.')]
        base_s = [p for n, p in ms.named_parameters() if n != 's']
        db = max(float((a - b).abs().max()) for a, b in zip(base_ref, base_b))
        dsl = max(float((a - b).abs().max()) for a, b in zip(base_ref, base_s))

    x, meta, _ = E.data(11 * 10000, 64, 0, 80)
    with torch.no_grad():
        k_init = {}
        for k in ('single', 'burst5', 'decay-slow'):
            torch.manual_seed(seed)
            m = A.BankAudit(.9, learnable=False, init=k)
            k_init[k] = float((m.mixed_kernel(x, meta)[:, 0] -
                               A.FIXED_KERNEL[k]).abs().max())
        lam_pts = {'lam0': float((ms.kernel_a(0.0) - A.FIXED_KERNEL['single']).abs().max()),
                   'lam1': float((ms.kernel_a(1.0) - A.FIXED_KERNEL['burst5']).abs().max()),
                   'lam.85': float((ms.kernel_a(0.85) - A.FIXED_KERNEL['decay-slow']).abs().max())}
    return {'base_maxdiff_bank': db, 'base_maxdiff_shared': dsl,
            'bank_init_kernel_maxdiff': k_init, 'shared_lambda_limits': lam_pts,
            'shared_init_lambda': round(float(ms.lam().detach()), 6)}


def energy_check():
    seed = 11
    torch.manual_seed(seed)
    ref = E.Flow(.9, learnable=False, kernel_name='single')
    torch.manual_seed(seed)
    mb = A.BankAudit(.9, learnable=False, init='single')
    torch.manual_seed(seed)
    ms = A.SharedLambda(.9, learnable=False, lam0=0.05)
    x = torch.zeros(4, 40, 2)
    x[:, 10, 0] = torch.tensor([1., -2., .5, 1.5])
    meta = torch.zeros(4, 40, 7)
    meta[:, :, 2] = torch.linspace(0, 1, 40)
    with torch.no_grad():
        for m in (ref, mb, ms):
            E.make_fixed_B(m, seed, 0.0)
        e_ref = float(((x @ ref.B).reshape(4, 40, 4, E.M).pow(2).sum()))
        e_bank = float(mb.writes(x, meta).pow(2).sum())
        e_sh = float(ms.writes(x, meta).pow(2).sum())
    return {'E_fixed_single': e_ref, 'rel_bank': abs(e_bank - e_ref) / e_ref,
            'rel_shared': abs(e_sh - e_ref) / e_ref}


def grad_check():
    seed, T = 11, 80
    tr = E.data(seed * 10000, 128, 0, T)
    out = {}
    torch.manual_seed(seed)
    mb = A.BankAudit(.9, learnable=False, init='single')
    E.make_fixed_B(mb, seed, 0.0)
    fp = [p for n, p in mb.named_parameters() if n.startswith('wbank.')]
    mb.train()
    mb.detach_window = True
    mb.zero_grad(set_to_none=True)
    F.binary_cross_entropy_with_logits(mb(tr[0], tr[1]), tr[2]).backward()
    out['bank_trunc'] = float(sum((p.grad.norm() ** 2 if p.grad is not None else 0.0)
                                  for p in fp) ** .5)
    mb.zero_grad(set_to_none=True)
    mb.eval()
    mb.detach_window = False
    l = F.binary_cross_entropy_with_logits(mb(tr[0], tr[1]), tr[2])
    out['bank_full'] = float(sum(g.norm() ** 2 for g in torch.autograd.grad(l, fp)))

    torch.manual_seed(seed)
    ms = A.SharedLambda(.9, learnable=False, lam0=0.05)
    E.make_fixed_B(ms, seed, 0.0)
    ms.train()
    ms.detach_window = True
    ms.zero_grad(set_to_none=True)
    F.binary_cross_entropy_with_logits(ms(tr[0], tr[1]), tr[2]).backward()
    out['shared_trunc'] = float(ms.s.grad.norm()) if ms.s.grad is not None else 0.0
    ms.zero_grad(set_to_none=True)
    ms.eval()
    ms.detach_window = False
    l = F.binary_cross_entropy_with_logits(ms(tr[0], tr[1]), tr[2])
    out['shared_full'] = float(torch.autograd.grad(l, [ms.s])[0].norm())
    return out


if __name__ == '__main__':
    print(json.dumps({'init': init_check(), 'energy': energy_check(), 'grad': grad_check()}, indent=1))

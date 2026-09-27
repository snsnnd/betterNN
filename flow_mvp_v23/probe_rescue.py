"""V23 诊断：K=5 失败流能否被 Full core 微调救回（可达性 vs 优化/平台）。

对 T=20 xor 的 K=5 流：先按主协议训 500 epochs（B Full / core K=5），
再用同一模型继续 100 epochs 的 B Full / core Full 微调，比较 y0。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import experiment as E  # noqa: E402

torch.set_num_threads(1)


def train(m, tr, gen, core, opt_core, opt_b, named, full_names, full_params, K, epochs):
    for _ in range(epochs):
        m.train()
        if K != 'full':
            m.detach_period = int(K)
        order = torch.randperm(512, generator=gen)
        for ch in order.split(E.BATCH):
            xb, mb, yb = tr[0][ch], tr[1][ch], tr[2][ch]
            opt_core.zero_grad(set_to_none=True)
            opt_b.zero_grad(set_to_none=True)
            if K == 'full':
                m.detach_window = False
                F.binary_cross_entropy_with_logits(m(xb, mb), yb).backward()
                m.detach_window = True
            else:
                m.detach_window = True
                F.binary_cross_entropy_with_logits(m(xb, mb), yb).backward()
                gt = {n: p.grad.detach().clone() for n, p in named if p.grad is not None}
                for p in m.parameters():
                    p.grad = None
                m.eval()
                m.detach_window = False
                lf = F.binary_cross_entropy_with_logits(m(xb, mb), yb)
                gf = torch.autograd.grad(lf, full_params)
                m.train()
                gi = 0
                for n, p in named:
                    if n in full_names:
                        p.grad = gf[gi]
                        gi += 1
                    else:
                        p.grad = gt[n]
            torch.nn.utils.clip_grad_norm_(core, 1.)
            torch.nn.utils.clip_grad_norm_([m.B_raw], 1.)
            opt_core.step()
            opt_b.step()


def probe(seed, task='xor', T=20, epochs=500, rescue_epochs=100):
    m = E.make_model(seed)
    tr = E.data_v23(seed * 10000, 512, task, T, 'transient')
    te = E.data_v23(seed * 10000 + 10, 1024, task, T, 'transient')
    core = [p for n, p in m.named_parameters() if n != 'B_raw']
    full_names = {n for n, _ in m.named_parameters() if n == 'B_raw' or E.group_of(n) == 'readout'}
    full_params = [p for n, p in m.named_parameters() if n in full_names]
    named = list(m.named_parameters())
    opt_core = torch.optim.Adam(core, lr=E.LR)
    opt_b = torch.optim.Adam([m.B_raw], lr=E.LR)
    gen = torch.Generator().manual_seed(seed + 5000)
    t0 = time.perf_counter()
    train(m, tr, gen, core, opt_core, opt_b, named, full_names, full_params, '5', epochs)
    a5 = E.evaluate(m, te)
    train(m, tr, gen, core, opt_core, opt_b, named, full_names, full_params, 'full', rescue_epochs)
    ar = E.evaluate(m, te)
    return {'seed': seed, 'K5': a5, 'rescue': ar, 'seconds': time.perf_counter() - t0}


if __name__ == '__main__':
    import multiprocessing as mp
    seeds = [int(s) for s in sys.argv[1:]] or [11, 22, 33, 44, 55]
    with mp.get_context('fork').Pool(min(3, len(seeds))) as pool:
        out = list(pool.imap_unordered(probe, seeds))
    out.sort(key=lambda r: r['seed'])
    for r in out:
        print(f"seed {r['seed']}: K5 acc={r['K5']['acc']:.3f} (y0 {r['K5']['y0']:.3f}) → "
              f"rescue Full {r['rescue']['acc']:.3f} (y0 {r['rescue']['y0']:.3f}) [{r['seconds']:.0f}s]", flush=True)
    (ROOT / 'results/rescue_probe.json').write_text(json.dumps(out, indent=2))
    print('mean K5', np.mean([r['K5']['acc'] for r in out]), 'mean rescue', np.mean([r['rescue']['acc'] for r in out]))

"""第十三轮：梯度冲突定位。在 v11 检查点上计算任务梯度的分组 pairwise cosine 与 alpha*。"""
import argparse, itertools, json, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from mechanism import build, data, group_params

ROOT = Path(__file__).parent
torch.set_num_threads(1)
GROUPS = ['W', 'write', 'hold', 'route', 'readout']


def grads_for(m, seed, task, n=256, truncate=True):
    tr = data(seed * 10000 + task * 100, 512, task)
    x, meta, y = tr[0][:n], tr[1][:n], tr[2][:n]
    (m.train() if truncate else m.eval())
    loss = F.binary_cross_entropy_with_logits(m(x, meta), y)
    params = group_params(m)
    out = {}
    keys = list(params)
    for i, k in enumerate(keys):
        gs = torch.autograd.grad(loss, params[k], retain_graph=(i < len(keys) - 1))
        out[k] = [g.detach().reshape(-1) for g in gs]
    return out


def flat(gs):
    return torch.cat(gs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v11', default='../flow_mvp_v11/results')
    ap.add_argument('--order', default='o0')
    ap.add_argument('--ratios', nargs='+', type=float, default=[0, .125])
    ap.add_argument('--seeds', nargs='+', type=int, default=[11, 22, 33, 44, 55])
    ap.add_argument('--stages', nargs='+', type=int, default=[1, 2, 3])
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--out', default='conflict.json')
    ap.add_argument('--modes', nargs='+', default=['truncated', 'full'])
    args = ap.parse_args()
    v11 = Path(args.v11) if Path(args.v11).is_absolute() else (ROOT / args.v11).resolve()
    labels = {0: '0', .125: '12.5'}
    order = [0, 1, 2, 3]
    records = []
    start = time.perf_counter()
    for mode in args.modes:
        for ratio in args.ratios:
            for seed in args.seeds:
                for stage in args.stages:
                    ck = torch.load(v11 / f"{args.order}_all_r{labels[ratio]}_{seed}_stage{stage}.pt",
                                    map_location='cpu', weights_only=False)
                    m = build(seed)
                    m.load_state_dict(ck['state'])
                    learned = order[:stage + 1]
                    grads = {t: grads_for(m, seed, t, args.batch, truncate=(mode == 'truncated')) for t in learned}
                    for i, j in itertools.combinations(learned, 2):
                        for g in GROUPS:
                            gi, gj = flat(grads[i][g]), flat(grads[j][g])
                            dot = float(torch.dot(gi, gj))
                            ni, nj = float(torch.dot(gi, gi)), float(torch.dot(gj, gj))
                            cos = dot / max(1e-12, (ni * nj) ** .5)
                            alpha_star = 0.0 if dot >= 0 else -dot / max(1e-12, ni - dot)
                            records.append({'order': args.order, 'ratio': ratio, 'label': labels[ratio], 'seed': seed,
                                            'stage': stage, 'old': i, 'new': j, 'is_newest': j == order[stage],
                                            'mode': mode, 'group': g, 'cos': cos, 'dot': dot,
                                            'n_old2': ni, 'n_new2': nj, 'alpha_star': alpha_star})
                    print(f"{mode} ratio={labels[ratio]} seed={seed} stage={stage} pairs={len(records)}", flush=True)
    (ROOT / args.out).write_text(json.dumps({'batch': args.batch, 'modes': args.modes, 'records': records,
                                             'seconds': time.perf_counter() - start}, indent=2))
    print('wrote', args.out, 'records', len(records), 'in', round(time.perf_counter() - start, 1), 's')


if __name__ == '__main__':
    main()

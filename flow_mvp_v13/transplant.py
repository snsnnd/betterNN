"""第十三轮：component transplant。最终状态分别换回 stage0 的 W/ctrl/readout 的 8 种组合。"""
import argparse, itertools, json, time
from pathlib import Path
import numpy as np
import torch
from mechanism import build, group_names, scores

ROOT = Path(__file__).parent
torch.set_num_threads(1)
COMPS = ['W', 'ctrl', 'readout']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v11', default='../flow_mvp_v11/results')
    ap.add_argument('--order', default='o0')
    ap.add_argument('--ratios', nargs='+', type=float, default=[0, .125])
    ap.add_argument('--seeds', nargs='+', type=int, default=[11, 22, 33, 44, 55])
    ap.add_argument('--out', default='transplant.json')
    args = ap.parse_args()
    v11 = Path(args.v11) if Path(args.v11).is_absolute() else (ROOT / args.v11).resolve()
    labels = {0: '0', .125: '12.5'}
    records = []
    start = time.perf_counter()
    for ratio in args.ratios:
        for seed in args.seeds:
            final = torch.load(v11 / f"{args.order}_all_r{labels[ratio]}_{seed}_stage3.pt",
                               map_location='cpu', weights_only=False)['state']
            old = torch.load(v11 / f"{args.order}_all_r{labels[ratio]}_{seed}_stage0.pt",
                             map_location='cpu', weights_only=False)['state']
            m = build(seed)
            names = group_names(m)
            for r in range(len(COMPS) + 1):
                for combo in itertools.combinations(COMPS, r):
                    sd = dict(final)
                    for comp in combo:
                        for n in names[comp]:
                            sd[n] = old[n]
                    m.load_state_dict(sd)
                    acc = scores(m, seed)
                    records.append({'order': args.order, 'ratio': ratio, 'label': labels[ratio], 'seed': seed,
                                    'combo': list(combo), 'n_old': len(combo), 'acc': acc})
            print(f"ratio={labels[ratio]} seed={seed} done", flush=True)
    (ROOT / args.out).write_text(json.dumps({'records': records, 'seconds': time.perf_counter() - start}, indent=2))
    print('wrote', args.out, 'records', len(records), 'in', round(time.perf_counter() - start, 1), 's')


if __name__ == '__main__':
    main()

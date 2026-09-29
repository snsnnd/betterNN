"""第二十六轮：Budget-Matched Temporal Write Re-baseline。

- 协议与 V24 Phase A 完全一致（Flow-v2、B fixed-disjoint、core K=5、Adam lr=.003、
  batch 128、数据/种子/RNG 不变），只换 write kernel（single/burst5/decay-slow）；
- 500 epochs；曲线：e=1、前 450 每 5 epoch、末 50 每 epoch 的 val，预算点加 test；
- Phase 0 回归：single@T80/task0/seed11/65ep 的 test@65 应与 V24 delay_single_T80_t0_11 逐位一致。
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).parent
# 以独立模块名加载 V24 experiment，避免与本文件同名自遮蔽
_spec = importlib.util.spec_from_file_location(
    'v24_experiment', ROOT.parent / 'flow_mvp_v24' / 'experiment.py')
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

torch.set_num_threads(int(os.environ.get('FLOW_THREADS', '1')))
torch.use_deterministic_algorithms(True)

KERNELS = ['single', 'burst5', 'decay-slow']
SEEDS = [11, 22, 33, 44, 55]
EPOCHS = 500
TEST_EPOCHS = (25, 50, 65, 100, 200, 300, 500)


def train_rebase(m, seed, task, T, epochs=EPOCHS):
    """与 E.train_fixed 相同的训练路径；只在 epoch 边界增加 eval（不消费 RNG）。"""
    tr = E.data(seed * 10000 + task * 100, 512, task, T)
    val = E.data(seed * 10000 + task * 100 + 1, 256, task, T)
    te = E.data(seed * 10000 + task * 100 + 10, 1024, task, T)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=E.LR)
    gen = torch.Generator().manual_seed(seed + task * 100 + 5000)
    curve = []
    for epoch in range(epochs):
        m.train()
        order = torch.randperm(512, generator=gen)
        for ch in order.split(E.BATCH):
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(m(tr[0][ch], tr[1][ch]), tr[2][ch])
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.)
            opt.step()
        e = epoch + 1
        if e % 5 == 0 or e == 1 or e >= epochs - 49 or e in TEST_EPOCHS:
            row = {'epoch': e, 'val': E.evaluate(m, val)['acc']}
            if e in TEST_EPOCHS:
                row['test'] = E.evaluate(m, te)['acc']
            curve.append(row)
    return curve


def run(kernel, T, task, seed, out, epochs=EPOCHS):
    name = f'rebase_{kernel}_T{T}_t{task}_{seed}'
    p = out / (name + '.json')
    if p.exists():
        return
    m = E.build(seed, .9, learnable=False, kernel_name=kernel)
    E.make_fixed_B(m, seed, 0.0)
    start = time.perf_counter()
    curve = train_rebase(m, seed, task, T, epochs)
    row = {'name': name, 'kernel': kernel, 'T': T, 'task': task, 'seed': seed,
           'epochs': epochs, 'curve': curve, 'lr': E.LR,
           'seconds': time.perf_counter() - start}
    p.write_text(json.dumps(row))
    test500 = next((c.get('test') for c in curve if c['epoch'] == epochs), None)
    print(f"{name}: test@{epochs}={test500} {row['seconds']:.0f}s", flush=True)


def _job(j):
    run(**j)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--T', type=int, default=80)
    ap.add_argument('--kernels', nargs='+', default=KERNELS)
    ap.add_argument('--tasks', nargs='+', type=int, default=[0, 1, 2, 3])
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--out', default=None)
    ap.add_argument('--jobs', type=int, default=1)
    args = ap.parse_args()
    out = ROOT / (args.out or f'results/rebase_{args.T}')
    out.mkdir(parents=True, exist_ok=True)
    jobs = [dict(kernel=k, T=args.T, task=t, seed=s, out=out, epochs=args.epochs)
            for k in args.kernels for t in args.tasks for s in args.seeds]
    print(f'rebase T={args.T} jobs={len(jobs)} epochs={args.epochs} out={out}', flush=True)
    t0 = time.perf_counter()
    if args.jobs <= 1:
        for j in jobs:
            _job(j)
    else:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(args.jobs) as pool:
            for _ in pool.imap_unordered(_job, jobs):
                pass
    print(f'all {len(jobs)} jobs in {time.perf_counter() - t0:.0f} s')


if __name__ == '__main__':
    main()

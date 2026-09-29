"""Phase 0 回归：train_rebase 前 65 epoch 必须与 V24 delay_* 同 seed 逐位一致。"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import experiment as R  # noqa: E402  (v26; 内部以 v24_experiment 加载 V24)

torch.set_num_threads(1)

SEED, TASK, T = 11, 0, 80


def main():
    out = {}
    for kernel in ('single', 'burst5', 'decay-slow'):
        m = R.E.build(SEED, .9, learnable=False, kernel_name=kernel)
        R.E.make_fixed_B(m, SEED, 0.0)
        curve = R.train_rebase(m, SEED, TASK, T, epochs=65)
        test65 = next(c['test'] for c in curve if c['epoch'] == 65)
        val65 = next(c['val'] for c in curve if c['epoch'] == 65)
        v24 = json.loads((ROOT.parent / 'flow_mvp_v24' / 'results' / 'delay' /
                          f'delay_{kernel}_T{T}_t{TASK}_{SEED}.json').read_text())
        out[kernel] = {'v26_test65': test65, 'v24_final': v24['final'],
                       'test_diff': abs(test65 - v24['final']),
                       'val65_diff': abs(val65 - v24['curve'][-1]['acc'])}
    print(json.dumps(out, indent=1))
    assert max(v['test_diff'] for v in out.values()) == 0, 'regression failed'


if __name__ == '__main__':
    main()

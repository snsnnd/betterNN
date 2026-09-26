"""CPU一致性检查：v8 learned_topk 应与第七轮 topk_256 同种子存档逐位一致。"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
out = ROOT / 'parity' / 'learned_topk_256_11.json'
arc = ROOT.parent / 'flow_mvp_v7' / 'results' / 'topk_256_11.json'
if not out.exists():
    subprocess.run([sys.executable, 'experiment.py', '--modes', 'learned_topk', '--seeds', '11',
                    '--device', 'cpu', '--out', 'parity'], cwd=ROOT, check=True)
a = json.loads(arc.read_text())
b = json.loads(out.read_text())
lines = [f"matrix_equal: {a['matrix'] == b['matrix']}",
         f"v7 final_mean={a['final_mean']:.6f} v8 final_mean={b['final_mean']:.6f}",
         f"v7 forgetting={a['forgetting']:.6f} v8 forgetting={b['forgetting']:.6f}"]
(ROOT / 'parity.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))

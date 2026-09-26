"""第十七轮：可并行性验证。C 的仿射扫描等价性 + D 第二遍扫描等价性 + 深度/耗时对比。"""
import json, time
from pathlib import Path
import torch

ROOT = Path(__file__).parent
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
from experiment import build, data, N

torch.manual_seed(0)


def affine_params(m, x, meta):
    """把 preroute 模型展开成逐步仿射映射 (M_t, c_t)：h_{t+1} = M_t h_t + c_t。"""
    w = (m.W * m.mask).reshape(4, m.m, 4, m.m)
    eye = torch.eye(N)
    Ms, cs = [], []
    with torch.no_grad():
        for t in range(x.shape[1]):
            ctl = meta[:, t]
            a = .2 * m.hold(ctl).sigmoid()
            pad = x.new_zeros(len(x), 2)
            q = torch.cat([ctl, x[:, t], pad], 1)
            gate = m.route(q).sigmoid().reshape(-1, 4, 4)
            A = (gate[:, :, None, :, None] * w[None]).reshape(len(x), N, N)
            ar = a.repeat_interleave(m.m, dim=1)
            M = (1 - ar)[:, :, None] * eye + ar[:, :, None] * A
            c = ar * (x[:, t] @ m.B)
            Ms.append(M)
            cs.append(c)
    return torch.stack(Ms), torch.stack(cs)  # [T,b,n,n], [T,b,n]


def seq_rollout(Ms, cs, h0):
    h = h0
    out = []
    for t in range(len(Ms)):
        h = Ms[t] @ h[:, :, None]
        h = h[:, :, 0] + cs[t]
        out.append(h)
    return torch.stack(out)


def scan_rollout(Ms, cs, h0):
    """Hillis-Steele 前缀组合：组合律 (M2,c2)∘(M1,c1)=(M2M1, M2c1+c2)。"""
    M = Ms.clone()
    c = cs.clone()
    T = M.shape[0]
    step = 1
    while step < T:
        M_new = M.clone()
        c_new = c.clone()
        M_new[step:] = M[step:] @ M[:-step]
        c_new[step:] = (M[step:] @ c[:-step].unsqueeze(-1)).squeeze(-1) + c[step:]
        M, c = M_new, c_new
        step *= 2
    h = (M[-1] @ h0[:, :, None])[:, :, 0] + c[-1]
    return h, M, c


def check_preroute():
    m = build(11, 'preroute')
    x, meta, _ = data(11 * 10000 + 10, 8, 0, steps=64)
    Ms, cs = affine_params(m, x, meta)
    h0 = torch.zeros(len(x), N)
    h_seq = seq_rollout(Ms, cs, h0)[-1]
    h_scan, _, _ = scan_rollout(Ms, cs, h0)
    diff = float((h_seq - h_scan).abs().max())
    m.eval()
    with torch.no_grad():
        _, h_model = m(x, meta, record=True)
    diff_model = float((h_model[:, -1].reshape(len(x), -1) - h_seq).abs().max())
    return {'seq_vs_scan_max_abs': diff, 'model_vs_seq_max_abs': diff_model}, (x, meta)


def check_iter_pass2():
    m = build(11, 'iter')
    m.eval()
    x, meta, _ = data(11 * 10000 + 10, 8, 0, steps=64)
    with torch.no_grad():
        # 第一遍与第二遍的门由同一个 route 模块产生；复刻两遍以提取第二遍的仿射参数
        w = (m.W * m.mask).reshape(4, m.m, 4, m.m)
        h1 = x.new_zeros(len(x), 4, m.m)
        h1s = []
        for t in range(x.shape[1]):
            ctl = meta[:, t]
            a = .2 * m.hold(ctl).sigmoid()
            pad = x.new_zeros(len(x), 2)
            q = torch.cat([ctl, x[:, t], pad], 1)
            gate = m.route(q).sigmoid().reshape(-1, 4, 4)
            rec = (torch.einsum('bsi,sidj->bsdj', h1, w) * gate[:, :, :, None]).sum(1)
            h1 = (1 - a[:, :, None]) * h1 + a[:, :, None] * (rec + (x[:, t] @ m.B).reshape(-1, 4, m.m))
            h1s.append(h1)
        Ms, cs = [], []
        for t in range(x.shape[1]):
            ctl = meta[:, t]
            a = .2 * m.hold(ctl).sigmoid()
            q = torch.cat([ctl, h1s[t].sum(-1) / m.m], 1)
            gate = m.route(q).sigmoid().reshape(-1, 4, 4)
            A = (gate[:, :, None, :, None] * w[None]).reshape(len(x), N, N)
            ar = a.repeat_interleave(m.m, dim=1)
            M = (1 - ar)[:, :, None] * torch.eye(N) + ar[:, :, None] * A
            Ms.append(M)
            cs.append(ar * (x[:, t] @ m.B))
        Ms, cs = torch.stack(Ms), torch.stack(cs)
        h0 = torch.zeros(len(x), N)
        h_seq2 = seq_rollout(Ms, cs, h0)[-1]
        h_scan2, _, _ = scan_rollout(Ms, cs, h0)
        _, h_model = m(x, meta, record=True)
        return {'pass2_seq_vs_scan': float((h_seq2 - h_scan2).abs().max()),
                'pass2_model_vs_scan': float((h_model[:, -1].reshape(len(x), -1) - h_scan2).abs().max())}


def timing():
    m = build(11, 'preroute')
    x, meta, _ = data(11 * 10000 + 10, 8, 0, steps=512)
    Ms, cs = affine_params(m, x, meta)
    h0 = torch.zeros(len(x), N)
    t0 = time.perf_counter(); seq_rollout(Ms, cs, h0); t_seq = time.perf_counter() - t0
    t0 = time.perf_counter(); scan_rollout(Ms, cs, h0); t_scan = time.perf_counter() - t0
    return {'T': 512, 'batch': 8, 'seq_seconds': t_seq, 'scan_seconds': t_scan, 'speedup': t_seq / t_scan}


def main():
    import json as _json
    r1, _ = check_preroute()
    r2 = check_iter_pass2()
    r3 = timing()
    out = {'preroute': r1, 'iter_pass2': r2, 'timing': r3}
    (ROOT / 'scan_check.json').write_text(_json.dumps(out, indent=2))
    print(_json.dumps(out, indent=2))


if __name__ == '__main__':
    main()

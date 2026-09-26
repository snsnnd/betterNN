"""重载55个新模型并独立重算；核验拓扑控制和新任务标签。"""
import argparse,json
from pathlib import Path
import torch
from experiments import *

def checks():
    # 一次只改mask，完整候选权重来自同一生成器；B/C/控制器/读出必须一致。
    a=TopologyFlow('dynamic',101,11);b=TopologyFlow('dynamic',101,22)
    for k,v in a.state_dict().items():
        if k not in ['W','mask']:assert torch.equal(v,b.state_dict()[k]),k
    assert not torch.equal(a.mask,b.mask)
    assert abs(torch.linalg.matrix_norm(a.W,2).item()-.9)<1e-5
    st=TopologyFlow('static',101,11)
    # 静态Route输出不依赖输入；它通过固定零输入实现。
    inp=torch.zeros(17,8);assert torch.equal(st.lower(inp)[0],st.lower(inp)[-1])
    for n in [64]:
        (x,q,keys,noisy),y,info=salience_data(n,777)
        derived=(q[:,None]*keys).sum(-1);assert torch.equal(derived,info['valid'])
        truth=((x*derived[:,:,None]*q[:,None]).sum((1,2))>0).float();assert torch.equal(truth,y)
        (x,meta),y,info=delay_data(n,778)
        base=torch.stack([x[:,0,0]>0,x[:,1,1]>0],1)
        expected=torch.where(info['rule'][:,None].bool(),base.flip(1),base)
        assert torch.equal(expected,y.bool())
        for i in range(n):assert meta[i,:info['when'][i],3:5].sum()==0
    print('PASS: controlled masks share other parameters; spectral norm; content-derived relevance; delayed rules do not leak; oracle labels.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-only',action='store_true');args=ap.parse_args();torch.set_num_threads(1);checks()
    if args.data_only:return
    count=0;models=0
    for study in ['topology','salience','delay','delay_block']:
        root=Path('results')/study;rows=json.loads((root/'metrics.json').read_text());assert len(rows)==len(jobs_for(study))
        for r in rows:
            job={k:r[k] for k in ['study','mode','seed']+(['mask_seed'] if study=='topology' else [])};model=build(job);initial={k:v.clone() for k,v in model.state_dict().items()}
            ck=torch.load(root/(r['run_id']+'.pt'),weights_only=True,map_location='cpu');model.load_state_dict(ck['state_dict']);model.eval();verify_frozen(model,initial)
            _,_,tests=datasets(job)
            for name,data in tests.items():
                assert score(model,data)==r[name],(r['run_id'],name);count+=1
            for name,value in diagnostics(model,job,tests['id']).items():assert abs(value-r[name])<1e-7,(r['run_id'],name)
            models+=1
    print(f'PASS: {models} checkpoints, {count} primary scores and all saved diagnostic scores exactly reproduced; frozen weights unchanged.')
if __name__=='__main__':main()

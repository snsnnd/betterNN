"""已有五种子：配对、结构描述、路由软门诊断与静态替换。"""
import json,itertools
from pathlib import Path
import numpy as np
import torch
import networkx as nx
from scipy.stats import spearmanr
from legacy_experiment import build,dataset

@torch.no_grad()
def topology(W,mask):
    w=W.cpu().numpy();m=mask.cpu().numpy();G=nx.from_numpy_array(m,create_using=nx.DiGraph)
    no_loop=G.copy();no_loop.remove_edges_from(nx.selfloop_edges(no_loop))
    lengths=[d for u,dist in nx.all_pairs_shortest_path_length(no_loop) for v,d in dist.items() if u!=v]
    sv=np.linalg.svd(w,compute_uv=False);p=sv/sv.sum();eig=np.linalg.eigvals(w)
    return {'edge_count':int(m.sum()),'density_including_self':float(m.mean()),'mean_out_degree':float(m.sum(1).mean()),'mean_reachable_path':float(np.mean(lengths)),'reachable_pair_fraction':len(lengths)/(64*63),'strong_components':nx.number_strongly_connected_components(no_loop),'undirected_clustering':nx.average_clustering(no_loop.to_undirected()),'spectral_radius':float(np.abs(eig).max()),'spectral_norm':float(sv[0]),'effective_rank':float(np.exp(-(p[p>0]*np.log(p[p>0])).sum()))}

@torch.no_grad()
def forward_trace(model,data,gate_override=None):
    x,q,m,y=data;h=x.new_zeros(len(x),64);gg=[]
    for t in range(x.shape[1]):
        meta=torch.cat([q,x.new_full((len(x),1),t/(x.shape[1]-1)),m[:,t:t+1]],1)
        p=model.upper(meta).sigmoid();a=.2*model.update(meta).sigmoid();loc=h.reshape(-1,4,16).mean(-1)
        g=model.lower(torch.cat([q,loc],1)).sigmoid()
        if gate_override is not None:g=gate_override.expand(len(x),4) if gate_override.shape[0]==1 else gate_override[q.argmax(1)]
        h=(1-a)*h+a*torch.tanh((h*g.repeat_interleave(16,1))@(model.W*model.mask)+(x[:,t]*p)@model.B+meta@model.C);gg.append(g)
    return model.head(h).squeeze(-1),torch.stack(gg,1)

def main():
    torch.set_num_threads(1);out=Path('results');out.mkdir(exist_ok=True);rows=json.loads(Path('previous/metrics.json').read_text());index={(r['variant'],r['seed']):r for r in rows};records=[]
    for seed in [11,22,33,44,55]:
        model=build('g111');ck=torch.load(f'previous/g111_seed{seed}.pt',weights_only=True,map_location='cpu');model.load_state_dict(ck['state_dict']);model.eval()
        data=dataset(4096,seed*1000+10);small=tuple(t[:512] for t in data)
        logits,g=forward_trace(model,small)
        with torch.no_grad():assert torch.equal(logits,model(*small[:3]))
        # 静态替换值只由训练集估计，避免用测试分布校准。
        training=dataset(3072,seed*1000);_,gt=forward_trace(model,training)
        global_mean=gt.mean((0,1))[None];task_mean=torch.stack([gt[training[1][:,k].bool()].mean((0,1)) for k in range(4)])
        deg=model.mask.sum(1).reshape(4,16).sum(1);weights=deg/deg.sum()
        weighted=lambda t:float((t*weights).sum(-1).mean())
        eps=1e-7;entropy=-(g.clamp(eps,1-eps)*g.clamp(eps,1-eps).log2()+(1-g).clamp(eps,1-eps)*(1-g).clamp(eps,1-eps).log2())
        # 同一信号及标记反事实更换q，测量任务依赖；不是目标准确率测试。
        probes=[]
        for k in range(4):
            q=torch.nn.functional.one_hot(torch.full((len(small[0]),),k),4).float();_,gk=forward_trace(model,(small[0],q,small[2],small[3]));probes.append(gk)
        dist=np.zeros((4,4))
        for i,j in itertools.combinations(range(4),2):dist[i,j]=dist[j,i]=float((probes[i]-probes[j]).abs().mean())
        r={'seed':seed,**topology(model.W,model.mask),'route_delta_id_pp':100*(index['g111',seed]['id12']-index['g101',seed]['id12']),'route_delta_delay_pp':100*(index['g111',seed]['delay24']-index['g101',seed]['delay24']), 'soft_mean_gate':weighted(g),'bernoulli_proxy_entropy_bits':weighted(entropy),'temporal_abs_change':float((g[:,1:]-g[:,:-1]).abs().mean()),'counterfactual_task_distance':float(dist[np.triu_indices(4,1)].mean()),'task_distance_matrix':dist.tolist()}
        for threshold in [.25,.5,.75]:r[f'edge_fraction_above_{threshold}']=weighted((g>threshold).float())
        for name,d in [('id',data),('delay',dataset(4096,seed*1000+13,steps=24))]:
            for override,gate in [('dynamic',None),('global_static',global_mean),('task_static',task_mean)]:
                pred,_=forward_trace(model,d,gate);r[f'{name}_{override}_accuracy']=float(((pred>0)==d[3].bool()).float().mean())
        records.append(r)
    (out/'old_route_analysis.json').write_text(json.dumps(records,indent=2))
    features=['mean_reachable_path','undirected_clustering','spectral_radius','effective_rank','edge_count']
    correlations={k:float(spearmanr([r[k] for r in records],[r['route_delta_id_pp'] for r in records]).statistic) for k in features}
    (out/'old_exploratory_correlations.json').write_text(json.dumps(correlations,indent=2))
    pairs=[]
    for seed in [11,22,33,44,55]:
        pairs.append({'seed':seed,**{v:{k:index[v,seed][k] for k in ['id12','shift12','delay24']} for v in ['g111','g101','g110','g011']}})
    (out/'old_paired_accuracy.json').write_text(json.dumps(pairs,indent=2))
    print('OLD_ANALYSIS_COMPLETE',json.dumps([{k:r[k] for k in ['seed','route_delta_id_pp','counterfactual_task_distance','temporal_abs_change','id_dynamic_accuracy','id_global_static_accuracy','id_task_static_accuracy']} for r in records]))
if __name__=='__main__':main()

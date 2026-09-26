"""复用第三轮检查点：严格配对、初始图指标、路由使用与干预。"""
import itertools,json,math
from pathlib import Path
import numpy as np
import networkx as nx
from scipy.stats import rankdata
import torch
from prior_v3 import build,dataset,accuracy

def graph_features(w,mask):
    n=len(mask);adj=mask.copy();np.fill_diagonal(adj,0);g=nx.from_numpy_array(adj,create_using=nx.DiGraph)
    dist=[d for source,dd in nx.all_pairs_shortest_path_length(g) for target,d in dd.items() if source!=target]
    ss=np.linalg.svd(w,compute_uv=False);p=ss/ss.sum();eig=np.linalg.eigvals(w)
    return {'nodes':n,'edges_excluding_self':g.number_of_edges(),'mean_out_degree':g.number_of_edges()/n,'reachable_pair_fraction':len(dist)/(n*(n-1)),'largest_scc_fraction':max(map(len,nx.strongly_connected_components(g)))/n,'mean_shortest_path_reachable':float(np.mean(dist)),'directed_clustering':float(nx.average_clustering(g)),'spectral_radius':float(np.max(np.abs(eig))),'spectral_norm':float(ss[0]),'effective_rank':float(np.exp(-np.sum(p*np.log(np.maximum(p,1e-15))))),'stable_rank':float((ss**2).sum()/ss[0]**2)}

@torch.no_grad()
def collect(model,x,q,m):
    trace=[];handle=model.lower.register_forward_hook(lambda module,args,output:trace.append(output.sigmoid().detach()))
    model(x,q,m);handle.remove();return torch.stack(trace,1)

@torch.no_grad()
def intervene(model,data,replacement):
    # replacement是训练校准集均值的logit，前向hook替换下层输出。
    def hook(module,args,output):
        if replacement.ndim==1:return replacement[None,:].expand_as(output)
        q=args[0][:,:4].argmax(1);return replacement[q]
    h=model.lower.register_forward_hook(hook)
    result=accuracy(model,data);h.remove();return result

def main():
    torch.set_num_threads(1);out=Path('results');out.mkdir(exist_ok=True)
    rows=json.loads(Path('prior/v3_metrics.json').read_text());index={(r['seed'],r['variant']):r for r in rows};seeds=[11,22,33,44,55]
    paired=[];features=[];util=[]
    for seed in seeds:
        full=index[seed,'g111']
        for other in ['g101','g110','g011']:
            paired.append({'seed':seed,'full':'g111','comparison':other,**{f'{c}_delta_pp':100*(full[c]-index[seed,other][c]) for c in ['id12','shift12','random12','delay24','noise12']}})
        ck=torch.load(f'prior/g111_seed{seed}.pt',map_location='cpu',weights_only=True);model=build('g111');model.load_state_dict(ck['state_dict']);model.eval()
        mask=model.mask.numpy();w=model.W.detach().numpy();feat=graph_features(w,mask);feat.update({'seed':seed,'route_delta_id_pp':100*(full['id12']-index[seed,'g101']['id12'])});features.append(feat)
        x,q,m,y=dataset(256,seed*1000+10)
        # 相同信号/标记配四种查询，避免任务数据分布差异造成伪距离。
        xx=x.repeat(4,1,1);mm=m.repeat(4,1);qq=torch.eye(4).repeat_interleave(256,0)
        gates=collect(model,xx,qq,mm).reshape(4,256,12,4)
        edge_count=torch.tensor(mask.reshape(4,16,64).sum((1,2)));weights=edge_count/edge_count.sum()
        flat=gates.flatten(0,2);safe=flat.clamp(1e-7,1-1e-7)
        entropy=-(safe*safe.log2()+(1-safe)*(1-safe).log2())
        u={'seed':seed,'binary_gate_entropy_bits':float((entropy*weights).sum(-1).mean()),'temporal_gate_std':float((gates.std(2)*weights).sum(-1).mean())}
        for threshold in [.1,.5,.9]:u[f'edge_fraction_gate_gt_{threshold}']=float(((flat>threshold).float()*weights).sum(-1).mean())
        distances=[]
        for a,b in itertools.combinations(range(4),2):distances.append(float(((gates[a]-gates[b]).abs()*weights).sum(-1).mean()))
        u['paired_task_gate_distance']=float(np.mean(distances));u['task_mean_gate_patterns']=gates.mean((1,2)).tolist()
        # 校准只用训练样本；全球静态/按任务静态对照均是推理干预，非重训。
        tr=dataset(512,seed*1000);trg=collect(model,*tr[:3]);global_mean=trg.mean((0,1)).clamp(1e-6,1-1e-6)
        task_mean=torch.stack([trg[tr[1][:,k].bool()].mean((0,1)) for k in range(4)]).clamp(1e-6,1-1e-6)
        test=dataset(4096,seed*1000+10)
        u['original_id_accuracy']=accuracy(model,test)
        u['global_static_intervention_accuracy']=intervene(model,test,torch.logit(global_mean))
        u['task_static_intervention_accuracy']=intervene(model,test,torch.logit(task_mean))
        util.append(u)
    correlations={};delta=np.array([f['route_delta_id_pp'] for f in features])
    for key in features[0]:
        if key in ['seed','route_delta_id_pp']:continue
        x=np.array([f[key] for f in features]);correlations[key]=None if np.ptp(x)<1e-6*max(1.,float(np.abs(x).mean())) else {'pearson_r':float(np.corrcoef(x,delta)[0,1]),'spearman_r':float(np.corrcoef(rankdata(x),rankdata(delta))[0,1])}
    for name,value in [('prior_paired',paired),('topology_features',features),('topology_correlations_exploratory',correlations),('prior_route_utilization',util)]:
        (out/f'{name}.json').write_text(json.dumps(value,indent=2))
    print('AUDIT COMPLETE',flush=True)
    print(json.dumps({'route_id_deltas_pp':[f['route_delta_id_pp'] for f in features],'mean_task_distance':float(np.mean([u['paired_task_gate_distance'] for u in util])),'dynamic_accuracy':float(np.mean([u['original_id_accuracy'] for u in util])),'global_static_accuracy':float(np.mean([u['global_static_intervention_accuracy'] for u in util])),'task_static_accuracy':float(np.mean([u['task_static_intervention_accuracy'] for u in util]))},indent=2))
if __name__=='__main__':main()

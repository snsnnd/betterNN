import copy,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from base_routing import RoutedNet,RoutingGRU,dataset,evaluate
ROOT=Path(__file__).parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
SEEDS=[11,22,33];KINDS=['frozen','slow','joint','gru'];EPOCHS=60
# A swaps, B duplicates selected source, C duplicates other source, D reverse swaps.
def data(seed,n,task):
    x,old,y,ab,r=dataset(n,seed,20,4)
    meta=torch.zeros(n,20,7);meta[:,:,:3]=old;meta[:,4:,3+task]=1
    idx=torch.tensor([[[0,1],[1,0]],[[0,0],[1,1]],[[1,1],[0,0]],[[1,0],[0,1]]])[task,r]
    y=(ab.gather(1,idx)>0).float()
    return x,meta,y,ab,r

def initial(kind,seed):
    torch.manual_seed(seed)
    if kind=='gru':
        m=RoutingGRU();old=torch.load(ROOT/'initial'/f'gru_{seed}_0.pt',weights_only=False)['state'];m.load_state_dict(old)
        m.gru=nn.GRU(9,32,batch_first=True)
        state=m.state_dict()
        for name,p in old.items():
            if name=='gru.weight_ih_l0':state[name].zero_();state[name][:,:5]=p
            else:state[name]=p
        m.load_state_dict(state)
    else:
        m=RoutedNet('dynamic_slow');old=torch.load(ROOT/'initial'/f'joint_{seed}_0.pt',weights_only=False)['state'];m.load_state_dict(old)
        m.write[0]=nn.Linear(7,8);m.hold=nn.Linear(7,4);m.route[0]=nn.Linear(11,16)
        state=m.state_dict()
        for name,p in old.items():
            if name in ['write.0.weight','hold.weight']:
                state[name].zero_();state[name][:,:3]=p
            elif name=='route.0.weight':
                state[name].zero_();state[name][:,:3]=p[:,:3];state[name][:,7:]=p[:,3:]
            else:state[name]=p
        m.load_state_dict(state);m.W.requires_grad_(kind!='frozen')
    return m

def optimizer(m,kind):
    groups=[{'params':[p for n,p in m.named_parameters() if p.requires_grad and n!='W'],'lr':.003}]
    if kind in ['slow','joint']:groups.append({'params':[m.W],'lr':.00015 if kind=='slow' else .003})
    return torch.optim.Adam(groups)

def learn(m,kind,seed,task):
    train=data(seed*10000+task*100,1024,task)
    opt=optimizer(m,kind);gen=torch.Generator().manual_seed(seed+task*100+5000)
    hist=[]
    for epoch in range(EPOCHS):
        m.train();total=0
        for ids in torch.randperm(1024,generator=gen).split(256):
            opt.zero_grad(set_to_none=True);loss=F.binary_cross_entropy_with_logits(m(train[0][ids],train[1][ids]),train[2][ids]);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();total+=loss.detach().item()*len(ids)
        m.eval();val=data(seed*10000+task*100+1,512,task)
        hist.append({'loss':total/1024,'validation':evaluate(m,val)['pair_accuracy']})
    # Fixed last-epoch state; never select based on old-task tests.
    return hist

def scores(m,seed):
    m.eval();return [evaluate(m,data(seed*10000+t*100+10,2048,t))['pair_accuracy'] for t in range(4)]

def main():
    config={'seeds':SEEDS,'kinds':KINDS,'epochs_per_stage':EPOCHS,'train':1024,'validation':512,'test_per_task':2048,'steps':20,'rule_arrival':4,'lr':.003,'slow_W_lr':.00015,'order':['A','B','C','D'],'initial':'v5 full-BPTT joint Flow or GRU','replay':False,'checkpoint':'fixed final epoch','optimizer':'reset each new task for all groups','torch':torch.__version__}
    (OUT/'config.json').write_text(json.dumps(config,indent=2));rows=[]
    for seed in SEEDS:
      for kind in KINDS:
        m=initial(kind,seed);before=copy.deepcopy(m.state_dict());matrix=[scores(m,seed)];histories={}
        # A has already been learned in v5. Continue B,C,D with explicit task ID.
        for task in [1,2,3]:
            histories[str(task)]=learn(m,kind,seed,task);matrix.append(scores(m,seed))
            if kind!='gru':
                for key in ['B','mask']+(['W'] if kind=='frozen' else []):assert torch.equal(before[key],m.state_dict()[key])
            torch.save({'state':m.state_dict(),'seed':seed,'kind':kind,'stage':task},OUT/f'{kind}_{seed}_stage{task}.pt')
            print(kind,seed,'stage',task,[round(x,3) for x in matrix[-1]],flush=True)
        diag=[matrix[t][t] for t in range(4)]
        row={'seed':seed,'kind':kind,'matrix':matrix,'acquisition':diag,'final_mean':float(np.mean(matrix[-1])),'forgetting_mean_old':float(np.mean([max(matrix[s][t] for s in range(t,4))-matrix[-1][t] for t in range(3)])),'histories':histories}
        rows.append(row);(OUT/'metrics.json').write_text(json.dumps(rows,indent=2))
    # Independent adaptation controls, same pretrained starting point: avoid blaming task impossibility on forgetting.
    controls=[]
    for seed in SEEDS:
      for task in [1,2,3]:
        m=initial('joint',seed);hh=learn(m,'joint',seed,task);sc=scores(m,seed)
        controls.append({'seed':seed,'task':task,'score':sc[task],'all_scores':sc,'history':hh})
        torch.save({'state':m.state_dict(),'seed':seed,'kind':'joint','stage':task},OUT/f'control_{seed}_{task}.pt')
        (OUT/'controls.json').write_text(json.dumps(controls,indent=2));print('independent',seed,task,round(sc[task],3),flush=True)
    (OUT/'DONE').write_text('12 continual streams, 36 adaptation stages, 9 independent controls completed')
if __name__=='__main__':main()

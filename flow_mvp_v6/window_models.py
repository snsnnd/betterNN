import copy,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from base_routing import RoutedNet,RoutingGRU,dataset,evaluate

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
OUT=Path(__file__).parent/'attribution';OUT.mkdir(exist_ok=True)
# Identical recurrence as v4, with explicit boundaries; eval never truncates.
import inspect
src=inspect.getsource(RoutedNet.forward)
src=src.replace('    def forward','def forward',1)
lines=src.splitlines();src='\n'.join([lines[0]]+[s[4:] for s in lines[1:]])
src=src.replace('ctl=meta[:,t];','if self.training and self.window and t and t % self.window == 0: h=h.detach()\n        ctl=meta[:,t];')
space={'torch':torch};exec(src,space)
class Flow(RoutedNet):
    forward=space['forward']
    def __init__(self,kind,window):
        super().__init__('dynamic_slow' if kind!='frozen' else 'dynamic');self.window=window
class GRU(RoutingGRU):
    def __init__(self,window):super().__init__();self.window=window
    def forward(self,x,meta):
        z=torch.cat([x,meta],-1);h=None
        k=self.window if self.training and self.window else z.shape[1]
        for start in range(0,z.shape[1],k):
            out,h=self.gru(z[:,start:start+k],h)
            if start+k<z.shape[1]:h=h.detach()
        return self.head(out[:,-1])
def build(kind,window):return GRU(window) if kind=='gru' else Flow(kind,window)
def data(seed,n,steps=20):return dataset(n,seed,steps,4)

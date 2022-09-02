from .graph import GNN_Net_Graph
from .rgcn import RGCN_Net_Graph
from .rgin import RGIN_Net_Graph
from .gine import GINE_Net_Graph
from functools import partial

def get_model_cls(model_cls):
    if model_cls == "rgcn":
        return RGCN_Net_Graph
    if model_cls == 'rgin':
        return RGIN_Net_Graph
    if model_cls == "gine":
        return GINE_Net_Graph
    return partial(GNN_Net_Graph, gnn=model_cls)

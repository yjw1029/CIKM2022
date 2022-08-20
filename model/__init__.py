from .graph import GNN_Net_Graph
from .rgcn import RGCN_Net_Graph

def get_model_cls(model_cls):
    if model_cls == "rgcn":
        return RGCN_Net_Graph
    return GNN_Net_Graph

from .base import BaseAgg
from .non_uniform import NonUniformAgg
from .fednova import FedNovaAgg

def get_agg_cls(agg_cls):
    '''
    Args: 
        agg_cls: selected aggregation model
        
    Return:
        eval(agg_cls): corresponding aggregation model
    '''
    return eval(agg_cls)
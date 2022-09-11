from .base import BaseAgg
from .non_uniform import NonUniformAgg
from .fednova import FedNovaAgg
from .att_agg import AttAgg

def get_agg_cls(agg_cls):
    return eval(agg_cls)
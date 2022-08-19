from .base import BaseAgg
from .non_uniform import NonUniformAgg

def get_agg_cls(agg_cls):
    return eval(agg_cls)
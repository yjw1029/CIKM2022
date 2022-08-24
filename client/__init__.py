from .base import BaseClient
from .rgcn import RGCNClient
from .rgin import RGINClient

def get_client_cls(client_cls):
    return eval(client_cls)
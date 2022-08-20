from .base import BaseClient
from .rgcn import RGCNClient

def get_client_cls(client_cls):
    return eval(client_cls)
from .base import BaseClient
def get_client_cls(client_cls):
    return eval(client_cls)
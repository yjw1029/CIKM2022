from .base import BaseServer

def get_server_cls(server_cls):
    return eval(server_cls)
from .base import BaseServer

def get_server_cls(server_cls):
    '''
    Args: 
        server_cls: selected server model type
        
    Return:
        eval(server_cls): corresponding server model
    '''
    return eval(server_cls)
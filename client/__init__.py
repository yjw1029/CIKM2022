from .base import BaseClient
from .rgnn import RGNNClient
from .flreco import FLRecoClient, FLRecoRGNNClient

def get_client_cls(client_cls):
    '''
    Args: 
        client_cls: selected client model type
        
    Return:
        eval(client_cls): corresponding client model
    '''
    return eval(client_cls)
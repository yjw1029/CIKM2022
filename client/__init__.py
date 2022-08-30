from .base import BaseClient
from .rgnn import RGNNClient
from .flreco import FLRecoClient, FLRecoRGNNClient

def get_client_cls(client_cls):
    return eval(client_cls)
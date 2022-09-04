from .base import BaseClient
from .rgnn import RGNNClient
from .flreco import FLRecoClient, FLRecoRGNNClient
from .pretrain import PretrainClient

def get_client_cls(client_cls):
    return eval(client_cls)
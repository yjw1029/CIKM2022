from .local import LocalTrainer
from .fedavg import FedAvgTrainer

def get_trainer_cls(trainer_cls):
    '''
    Args: 
        trainer_cls: selected trainer model type
        
    Return:
        eval(trainer_cls): corresponding trainer model
    '''
    return eval(trainer_cls)
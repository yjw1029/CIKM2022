from .local import LocalTrainer
from .fedavg import FedAvgTrainer
from .kfold import KFoldLocalTrainer

def get_trainer_cls(trainer_cls):
    return eval(trainer_cls)
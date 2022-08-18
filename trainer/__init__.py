from .local import LocalTrainer

def get_trainer_cls(trainer_cls):
    return eval(trainer_cls)
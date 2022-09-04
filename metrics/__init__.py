from .error_rate import ErrorRate
from .mse import MSE
from .count import Count

def get_metric(name):
    if name == "mse":
        return MSE
    if name == "error_rate":
        return ErrorRate
    if 'loss' in name:
        return Count
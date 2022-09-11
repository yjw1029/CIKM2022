from .error_rate import ErrorRate
from .mse import MSE

def get_metric(name):
    '''
    Args: 
        name: selected metric type
        
    Return:
        corresponding metrics
    '''
    if name == "mse":
        return MSE
    if name == "error_rate":
        return ErrorRate
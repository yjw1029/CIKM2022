import logging
import sys
import torch


def setuplogger(args):
    '''
    set up logger
    '''
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(f"[%(levelname)s %(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)

def is_name_in_list(name, list):
    '''
    Args:
        name: parameter name
        list: name list
    Return:
        judge whether the name is in the list
    '''
    return any([pattern in name for pattern in list])

@torch.no_grad()
def grad_to_vector(model, server_model, filter_list=[]):
    '''
    Args:
        model: local model
        server_model: model in server
        filter_list: parameters are not shared.

    Return:
        the vector formed by model gradient
    '''
    vec = []
    server_state = server_model.state_dict()
    for name, param in model.named_parameters():
        if param.requires_grad and not is_name_in_list(name, filter_list):
            server_param = server_state[name]
            vec.append((server_param - param).detach().view(-1))
    return torch.cat(vec)


@torch.no_grad()
def vector_to_grad(vec, model, filter_list=[]):
    '''
    Args:
        vec: gradient vector
        model: model 
        filter_list: parameters are not shared.

    Return:
        None
    
    pass the gradients to model gradients
    '''
    pointer = 0
    for name, param in model.named_parameters():
        if param.requires_grad and not is_name_in_list(name, filter_list):
            num_param = param.numel()
            param.grad = vec[pointer : pointer + num_param].view_as(param).clone()
            pointer += num_param


@torch.no_grad()
def vector_to_dict(vec, model, filter_list=[]):
    '''
    Args:
        vec: gradient vector
        model: model 
        filter_list: parameters are not shared.

    Return:
        change gradient from vector forms to dict forms. 
    '''
    state_dict = {}
    pointer = 0
    for name, param in model.named_parameters():
        if param.requires_grad and not is_name_in_list(name, filter_list):
            num_param = param.numel()
            state_dict[name] = vec[pointer : pointer + num_param].view_as(param).clone()
            pointer += num_param
    return state_dict


@torch.no_grad()
def dict_to_vector(state_dict, model, filter_list=[]):
    '''
    Args:
        state_dict: gradient dict
        model: model 
        filter_list: parameters are not shared.

    Return:
        change gradient from dict forms to vector forms. 
    '''
    vec = []
    for name, param in model.named_parameters():
        if param.requires_grad and not is_name_in_list(name, filter_list):
            vec.append(state_dict[name].detach().view(-1))
    return torch.cat(vec)

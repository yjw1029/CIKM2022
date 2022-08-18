import functools
import numpy as np
import logging
import sys
import math
import torch
import random
import math


def setuplogger(args, log_dir):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(f"[%(levelname)s %(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)

    # disable logging to a file if enable wandb
    if not args.enable_wandb:
        fh = logging.FileHandler(log_dir / f"log.{args.wandb_run}.txt")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        root.addHandler(fh)


@torch.no_grad()
def grad_to_vector(model, server_model):
    vec = []
    for param, server_param in zip(model.parameters(), server_model.parameters()):
        if param.requires_grad:
            vec.append((server_param - param).detach().view(-1))
    return torch.cat(vec)


@torch.no_grad()
def vector_to_grad(vec, model):
    pointer = 0
    for param in model.parameters():
        if param.requires_grad:
            num_param = param.numel()
            param.grad = vec[pointer : pointer + num_param].view_as(param).clone()
            pointer += num_param

@torch.no_grad()
def vector_to_dict(vec, model):
    state_dict = {}
    pointer = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = param.numel()
            state_dict[name] = vec[pointer : pointer + num_param].view_as(param).clone()
            pointer += num_param
    return state_dict

@torch.no_grad()
def dict_to_vector(state_dict, model):
    vec = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            vec.append(state_dict[name].detach().view(-1))
    return torch.cat(vec)
    

def visualize_data_dist(alpha, user_num):
    import matplotlib.pyplot as plt

    class_num = 10
    class_p = np.array([1 / class_num for c in range(class_num)])
    s = np.random.dirichlet(class_p * alpha, user_num).transpose()

    for i in range(class_num):
        plt.barh(range(user_num), s[i], left=None if i == 0 else np.sum(s[:i], axis=0))

    plt.savefig(f"../figure/{alpha}.png")


def get_client_sampler(seed, list, k):
    g = random.Random(seed)
    return functools.partial(g.sample, list, k)

if __name__ == "__main__":
    for alpha in [1000000, 100, 10, 1.0, 0.1]:
        visualize_data_dist(alpha, 10)

import torch

from .base import BaseAgg
from utils import vector_to_grad

class NonUniformAgg(BaseAgg):
    def __init__(self, args, server_model):
        self.args = args
        self.server_model = server_model 

        optimizer_cls = eval(f"torch.optim.{self.args.global_optim_cls}")
        self.optimizer = optimizer_cls(self.server_model.parameters(), lr=self.args.global_lr)

    def aggregate(self, clients_rslts):
        self.optimizer.zero_grad()

        clients_updates = []
        clients_num = []
        for uid in clients_rslts:
            clients_updates.append(clients_rslts[uid]["update_vec"])
            clients_num.append(clients_rslts[uid]["train_num"])
        
        clients_updates = torch.stack(clients_updates, dim=0)
        clients_num = torch.FloatTensor(clients_num).to(clients_updates.device)

        clients_weight = clients_num / clients_num.sum()

        agg_update_vec = torch.mm(clients_weight.unsqueeze(0), clients_updates).squeeze(0)
        
        vector_to_grad(agg_update_vec, self.server_model, filter_list=self.args.param_filter_list)

        self.optimizer.step()
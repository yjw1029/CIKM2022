import torch

from .base import BaseAgg
from utils import vector_to_grad

class AttAgg(BaseAgg):
    def __init__(self, args, server_model):
        self.args = args
        self.server_model = server_model 

        optimizer_cls = eval(f"torch.optim.{self.args.global_optim_cls}")
        self.optimizer = optimizer_cls(self.server_model.parameters(), lr=self.args.global_lr)

    def att_agg(self,clients_updates,mask_clients_updates):
        coff = torch.mm(clients_updates.unsqueeze(0),mask_clients_updates.permute(1,0))
        coff = torch.nn.functional.softmax(coff,-1)
        agg = torch.mm(coff,mask_clients_updates).squeeze(0)
        return agg

    def aggregate(self, clients_rslts):
        self.optimizer.zero_grad()

        mask_clients_updates = []
        clients_updates = []
        clients_num = []
        for uid in clients_rslts:
            if uid in self.args.mask_clients:
                mask_clients_updates.append(clients_rslts[uid]["update_vec"])
            else:
                clients_updates=clients_rslts[uid]["update_vec"]
        
        mask_clients_updates = torch.stack(mask_clients_updates, dim=0)

        agg = self.att_agg(clients_updates,mask_clients_updates)

        agg_update_vec = (agg+clients_updates)/2
        
        vector_to_grad(agg_update_vec, self.server_model, filter_list=self.args.param_filter_list)

        self.optimizer.step()
import logging

from metrics import get_metric
from .base import BaseTrainer
from .fedavg import FedAvgTrainer
from utils import grad_to_vector
import os
import torch

class PretrainTrainer(FedAvgTrainer):
    def __init__(self, args):
        super(PretrainTrainer,self).__init__(args)
        self.best_model = {}
        self.best_rslt = {}
        self.best_epoch = {}
        for u_id in self.clients.keys():
            self.best_model[u_id] = None
            self.best_rslt[u_id] = None
            self.best_epoch[u_id] = 0
        # no server in local trainer

    def run(self):
        self.pretrain()
        

    def init_fintune_metrics(self):
        for u_id in self.clients.keys():
            self.clients[u_id].metric_cals = {}
            for metric in ['error_rate', 'relative_impr']:
                # compute relative_impr at the final step of evaluation
                if metric == "relative_impr":
                    self.clients[u_id].metric_cals[metric] = None
                else:
                    self.clients[u_id].metric_cals[metric] = get_metric(metric)()


    def pretrain(self):
        for step in range(self.args.max_steps):
            # all clients participant in
            server_state_dict = self.server.model.state_dict()

            for uid in self.args.clients:
                self.clients[uid].load_model(
                    server_state_dict, filter_list=self.args.param_filter_list
                )
                train_steps, train_num = self.clients[uid].train(reset_optim=True)
                update_vec = grad_to_vector(
                    self.clients[uid].model,
                    self.server.model,
                    filter_list=self.args.param_filter_list,
                )

                rslt = dict(
                    train_steps=train_steps, train_num=train_num, update_vec=update_vec
                )
                self.server.collect(uid, rslt)

            self.server.update(step=step)

            if step % self.args.eval_steps == 0:
                eval_rslt = self.evaluate_all_clients(step, load_server_model=True)
                for uid in self.args.clients:
                    if self.best_rslt[uid] is None or eval_rslt[uid][self.clients[uid].major_metric] < self.best_rslt[uid]:
                        self.best_rslt[uid] = eval_rslt[uid][self.clients[uid].major_metric]
                        self.best_model[uid] = self.clients[uid].model.state_dict()
                        self.best_epoch[uid] = step

        self.save_best_model(self.args.out_path)

    def save_best_model(self,path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "best_steps.csv"), "w") as file:
            for uid in self.args.clients:
                file.write(f'{uid}, best epoch:{self.best_epoch[uid]},best rslt:{self.best_rslt[uid]}\n')
                torch.save(self.best_model[uid],os.path.join(path, f'{uid}.pt'))



    def evaluate_all_clients(self, step, load_server_model=False):
        all_relative_impr = []
        
        server_state_dict = self.server.model.state_dict()

        eval_all = {}

        for uid in self.args.clients:
            if load_server_model:
                self.clients[uid].load_model(
                    server_state_dict, filter_list=self.args.param_filter_list
                )
            eval_rslt = self.clients[uid].eval()
            eval_all[uid] = eval_rslt
            if "relative_impr" in eval_rslt:
                all_relative_impr.append(eval_rslt["relative_impr"])

            eval_str = "; ".join(
                [f"{metric}: {value}" for metric, value in eval_rslt.items()]
            )
            logging.info(f"client_{uid} step {step}: {eval_str}")

            if "nan" in eval_str:
                logging.info(f"client_{uid} gets nan. Stop training")
                exit()
        
        if len(all_relative_impr) > 0:
            overall_impr = torch.mean(torch.stack(all_relative_impr))
            logging.info(f"step {step} overall relative_impr {overall_impr}")

        return eval_all

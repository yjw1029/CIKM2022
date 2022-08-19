import logging
import torch

from .base import BaseTrainer
from utils import grad_to_vector


class FedAvgTrainer(BaseTrainer):
    def __init__(self, args):
        self.args = args
        self.init_clients()
        self.init_server()

    def run(self):
        for step in range(self.args.max_steps):
            # all clients participant in
            server_state_dict = self.server.model.state_dict()

            for uid in range(1, self.args.clients_num + 1):
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
                self.evaluate_all_clients(step)

        self.save_predictions_all_clients()


    def evaluate_all_clients(self, step):
        all_relative_impr = []
        
        server_state_dict = self.server.model.state_dict()
        for uid in range(1, self.args.clients_num + 1):
            self.clients[uid].load_model(
                server_state_dict, filter_list=self.args.param_filter_list
            )
            eval_rslt = self.clients[uid].eval()

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

    def save_predictions_all_clients(self):
        for uid in range(1, self.args.clients_num+1):
            logging.info(f"[+] saving predictions...")
            self.clients[uid].save_prediction(self.args.out_path)
            logging.info(f"[-] finish saving predictions for client_{uid}")

    def finetune(self):
        pass

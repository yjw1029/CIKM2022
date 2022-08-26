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
        self.fl_training()

        if self.args.enable_finetune:
            self.finetune()
        else:
            self.save_predictions_all_clients()

    def fl_training(self):
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
                self.evaluate_all_clients(step, load_server_model=True)

        # distributed model to client and evaluate at the final step
        for uid in self.args.clients:
            self.clients[uid].load_model(
                server_state_dict, filter_list=self.args.param_filter_list
            )
        self.evaluate_all_clients(step)


    def evaluate_all_clients(self, step, load_server_model=False):
        all_relative_impr = []
        
        server_state_dict = self.server.model.state_dict()
        for uid in self.args.clients:
            if load_server_model:
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
        for uid in self.args.clients:
            logging.info(f"[+] saving predictions...")
            self.clients[uid].save_prediction(self.args.out_path)
            logging.info(f"[-] finish saving predictions for client_{uid}")


    def finetune(self):
        for uid in self.args.clients:
            best_rslt = None
            best_state_dict = None
            best_rslt_str = None
            no_imp_step = 0
            pre_rslt = 1e8

            for epoch in range(self.args.max_ft_steps):
                # local train 1 epoch
                self.clients[uid].finetune(reset_optim=False)
                eval_rslt = self.clients[uid].eval()

                eval_str = "; ".join([f"{metric}: {value}" for metric, value in eval_rslt.items()])
                logging.info(f"client_{uid} epoch {epoch}: {eval_str}")

                if best_rslt is None or eval_rslt[self.clients[uid].major_metric] < best_rslt:
                    best_rslt = eval_rslt[self.clients[uid].major_metric]
                    best_state_dict = self.clients[uid].model.state_dict()
                    best_rslt_str = eval_str
                
                if self.args.patient is not None:
                    no_imp_step += 1
                    if pre_rslt>eval_rslt[self.clients[uid].major_metric]:
                        no_imp_step=0
                    pre_rslt = eval_rslt[self.clients[uid].major_metric]
                    if self.args.patient < no_imp_step:
                        logging.info(f"[+] client_{uid} early stops due to worse performance than best result over {self.args.patient} steps")
                        break
            
            logging.info(f"[+] client_{uid} best rslt: {best_rslt_str}. saving predictions...")
            self.clients[uid].load_model(best_state_dict)
            self.clients[uid].save_prediction(self.args.out_path)
            self.clients[uid].save_best_rslt(uid, best_rslt_str, self.args.out_path)
            logging.info(f"[-] finish saving predictions for client_{uid}")
            del self.clients[uid]
import logging
import torch

from .base import BaseTrainer
from utils import grad_to_vector
import os


class FedAvgTrainer(BaseTrainer):
    def __init__(self, args):
        self.args = args

    def run(self):
        self.fl_training()

        if self.args.enable_finetune:
            self.finetune()
        else:
            self.save_predictions_all_clients()

    def fl_training(self):
        for val_fold in range(self.args.k_fold):
            self.args.val_fold = val_fold
            self.init_clients()
            self.init_server()
            self.best_model = {}
            self.best_rslt = {}
            self.best_epoch = {}
            for u_id in self.clients.keys():
                self.best_model[u_id] = None
                self.best_rslt[u_id] = None
                self.best_epoch[u_id] = 0

            logging.info(f"============ fold {val_fold} in all {self.args.k_fold} folds ===========")
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

            # distributed model to client and evaluate at the final step
            self.save_best_model(self.args.out_path,suffix=val_fold)
            self.save_predictions_all_clients(suffix=val_fold)

    def save_best_model(self,path,suffix=""):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f"eval_rslt{suffix}.txt"), "w") as file:
            for uid in self.args.clients:
                file.write(
                f"client {uid} best evaluation result: {(self.clients[uid].base_metric-self.best_rslt[uid])/self.clients[uid].base_metric} \n"
                )
                torch.save(self.best_model[uid],os.path.join(path, f'{uid}_{suffix}.pt'))


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


    def save_predictions_all_clients(self,suffix=""):
        for uid in self.args.clients:
            self.clients[uid].load_model(self.best_model[uid])
            logging.info(f"[+] saving predictions...")
            self.clients[uid].save_prediction(self.args.out_path,suffix=suffix)
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

                if best_rslt is None or eval_rslt[self.clients[uid].major_metric] <= best_rslt:
                    best_rslt = eval_rslt[self.clients[uid].major_metric]
                    best_state_dict = self.clients[uid].model.state_dict()
                    best_rslt_str = eval_str
                
                if self.args.patient is not None:
                    no_imp_step += 1
                    if pre_rslt >= eval_rslt[self.clients[uid].major_metric]:
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
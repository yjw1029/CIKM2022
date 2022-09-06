import logging

from metrics import get_metric
from .base import BaseTrainer
from .fedavg import FedAvgTrainer

class PretrainTrainer(FedAvgTrainer):
    def __init__(self, args):
        self.args = args
        self.init_clients()
        # no server in local trainer

    def run(self):
        self.pretrain()
        if self.args.enable_finetune:
            for u_id in self.clients.keys():
                self.clients[u_id].model.mode = "finetune"
                self.clients[u_id].major_metric = 'error_rate'
            self.init_fintune_metrics()
            self.finetune()
        else:
            self.save_predictions_all_clients()

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
        # local training clinet_by_client
        for uid in self.args.clients:
            best_rslt = None
            best_state_dict = None
            best_rslt_str = None
            no_imp_step = 0
            pre_rslt = 1e8
            for epoch in range(self.args.max_steps):
                # local train 1 epoch
                self.clients[uid].train(reset_optim=False)
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
            
            # logging.info(f"[+] client_{uid} best rslt: {best_rslt_str}. saving predictions...")
            # self.clients[uid].load_model(best_state_dict)
            # self.clients[uid].save_prediction(self.args.out_path)
            # self.clients[uid].save_best_rslt(uid, best_rslt_str, self.args.out_path)
            # logging.info(f"[-] finish saving predictions for client_{uid}")
            # del self.clients[uid]


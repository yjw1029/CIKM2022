import logging

from .base import BaseTrainer

class LocalTrainer(BaseTrainer):
    def __init__(self, args):
        self.args = args
        self.init_clients()
        # no server in local trainer

    def run(self):
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

                no_imp_step += 1
                if pre_rslt>eval_rslt[self.clients[uid].major_metric]:
                    no_imp_step=0
                pre_rslt = eval_rslt[self.clients[uid].major_metric]


                if best_rslt is None or eval_rslt[self.clients[uid].major_metric] < best_rslt:
                    best_rslt = eval_rslt[self.clients[uid].major_metric]
                    best_state_dict = self.clients[uid].model.state_dict()
                    best_rslt_str = eval_str
                
                if self.args.patient < no_imp_step:
                    logging.info(f"[+] client_{uid} early stops due to worse performance than best result over {self.args.patient} steps")
                    break
            
            logging.info(f"[+] client_{uid} best rslt: {best_rslt_str}. saving predictions...")
            self.clients[uid].load_model(best_state_dict)
            self.clients[uid].save_prediction(self.args.out_path)
            self.clients[uid].save_best_rslt(uid, best_rslt_str, self.args.out_path)
            logging.info(f"[-] finish saving predictions for client_{uid}")
            del self.clients[uid]


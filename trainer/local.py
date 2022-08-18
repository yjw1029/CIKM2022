import logging

from .base import BaseTrainer

class LocalTrainer(BaseTrainer):
    def __init__(self, args):
        self.args = args
        self.init_clients()
        # no server in local trainer

    def run(self):
        # local training clinet_by_client
        best_rslt = {}
        for uid in range(1, self.args.clients_num+1):
            best_rslt = None
            best_state_dict = None
            for epoch in range(self.args.max_steps):
                # local train 1 epoch
                self.clients[uid].train(reset_optim=False)
                eval_rslt = self.clients[uid].eval()

                eval_str = "; ".join([f"{metric}: {value}" for metric, value in eval_rslt.items()])
                logging.info(f"client_{uid} epoch {epoch}: {eval_str}")

                if best_rslt is None or eval_rslt[self.clients[uid].major_metric] < best_rslt:
                    best_rslt = eval_rslt[self.clients[uid].major_metric]
                    best_state_dict = self.clients[uid].model.state_dict()
            
            logging.info(f"[+] client_{uid} best rslt: {best_rslt}. saving predictions...")
            self.clients[uid].load_model(best_state_dict)
            self.clients[uid].save_prediction(self.args.out_path)
            logging.info(f"[-] finish saving predictions for client_{uid}")
            del self.clients[uid]


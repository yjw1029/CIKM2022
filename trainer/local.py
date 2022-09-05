import logging

from .base import BaseTrainer
from utils import EarlyStopper


class LocalTrainer(BaseTrainer):
    def __init__(self, args):
        # no server in local trainer
        self.args = args
        self.init_clients()

        self.early_stopper = EarlyStopper(self.args.patient)

    def run(self):
        # local training clinet_by_client
        for uid in self.args.clients:
            self.early_stopper.clear()
            for epoch in range(self.args.max_steps):
                # local train 1 epoch
                self.clients[uid].train(reset_optim=False)
                eval_rslt = self.clients[uid].eval()

                eval_str = "; ".join(
                    [f"{metric}: {value}" for metric, value in eval_rslt.items()]
                )
                logging.info(f"client_{uid} epoch {epoch}: {eval_str}")

                self.clients[uid].update_best(
                    eval_rslt=eval_rslt,
                    state_dict=self.clients[uid].model.state_dict(),
                    eval_str=eval_str,
                )

                need_stop = self.early_stopper.update(
                    eval_rslt[self.clients[uid].major_metric]
                )
                if need_stop:
                    logging.info(
                        f"[+] client_{uid} early stops due to worse performance than best result over {self.args.patient} steps"
                    )
                    break

            logging.info(
                f"[+] client_{uid} best rslt: {self.clients[uid].best_rslt_str}. saving checkpoints and predictions..."
            )
            self.clients[uid].load_model(self.clients[uid].best_state_dict)
            self.clients[uid].save_prediction(self.args.out_path, dataset="val")
            self.clients[uid].save_prediction(self.args.out_path, dataset="test")
            self.clients[uid].save_best_rslt(self.args.out_path)
            self.clients[uid].save_best_model(self.args.out_path)
            logging.info(f"[-] finish saving predictions for client_{uid}")
            del self.clients[uid]

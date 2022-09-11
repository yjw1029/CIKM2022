import logging

from .base import BaseTrainer
from utils import EarlyStopper


class LocalTrainer(BaseTrainer):
    """ Trainer for local optimization
    Args:
        args: Arguments
    """
    def __init__(self, args):
        # no server in local trainer
        self.args = args
        self.init_clients()

        self.early_stopper = EarlyStopper(self.args.patient)

    def run(self):
        ''' local training clinet_by_client

        For each client, conduct the local training and local evaluation for self.args.max_steps epochs.

        Args:
            self
        Returns:
            None
        '''
        for uid in self.args.clients:
            self.early_stopper.clear()
            for epoch in range(self.args.max_steps):
                # local train 1 epoch
                self.clients[uid].train(reset_optim=False)
                # local evaluation
                eval_rslt = self.clients[uid].eval()

                eval_str = "; ".join(
                    [f"{metric}: {value}" for metric, value in eval_rslt.items()]
                )
                logging.info(f"client_{uid} epoch {epoch}: {eval_str}")

<<<<<<< HEAD
                # update the best local result and cache the best model
                if best_rslt is None or eval_rslt[self.clients[uid].major_metric] <= best_rslt:
                    best_rslt = eval_rslt[self.clients[uid].major_metric]
                    best_state_dict = self.clients[uid].model.state_dict()
                    best_rslt_str = eval_str
                
                # early stop with patient
                if self.args.patient is not None:
                    no_imp_step += 1
                    if pre_rslt >= eval_rslt[self.clients[uid].major_metric]:
                        no_imp_step=0
                    pre_rslt = eval_rslt[self.clients[uid].major_metric]
                    if self.args.patient < no_imp_step:
                        logging.info(f"[+] client_{uid} early stops due to worse performance than best result over {self.args.patient} steps")
                        break
            # stop local training for defined epochs and save best result
            logging.info(f"[+] client_{uid} best rslt: {best_rslt_str}. saving predictions...")
            self.clients[uid].load_model(best_state_dict)
            self.clients[uid].save_prediction(self.args.out_path)
            self.clients[uid].save_best_rslt(uid, best_rslt_str, self.args.out_path)
=======
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
>>>>>>> 1e1790210fd197698c2cead4a53666a078b55711
            logging.info(f"[-] finish saving predictions for client_{uid}")
            # delete current client after finishing local training
            del self.clients[uid]

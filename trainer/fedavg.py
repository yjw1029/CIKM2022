import logging
import torch

from .base import BaseTrainer
from utils import grad_to_vector, EarlyStopper


class FedAvgTrainer(BaseTrainer):
    def __init__(self, args):
        self.args = args
        self.init_clients()
        self.init_server()

        self.early_stopper = EarlyStopper(self.args.patient)

    def run(self):
        """ Run FedAvg training

        Conduct FedAvg training and finetune local models if args.enable_finetune.
        Finally, save predictions for all clients.

        Args:
            self
        Returns:
            None
        """
        self.fl_training()

        if self.args.enable_finetune:
            self.finetune()
        else:
            self.save_all_clients()

    def fl_training(self):
        """ Conduct FedAvg training

        Iterate over self.args.max_steps rounds of FedAvg training with the following steps:
            Step 1. the server distributes the current global model to each client
            Step 2. each client conducts local training on their own devices
            Step 3. each client upload the local update cector to the server
            Step 4. the server collects local updates from all clients
            Step 5. the server updates the global model with aggregated update
        
        Evaluations are conducted for every args.eval_steps rounds.
        The final global model will be evaluated with loca data.

        Args:
            self
        Returns:
            None
        """
        self.early_stopper.clear()
        for step in range(self.args.max_steps):
            # all clients participant in
            server_state_dict = self.server.model.state_dict()

            for uid in self.args.clients:
                # Step 1
                self.clients[uid].load_model(
                    server_state_dict, filter_list=self.args.param_filter_list
                )
                # Step 2
                train_steps, train_num = self.clients[uid].train(reset_optim=True)
                # Step 3
                update_vec = grad_to_vector(
                    self.clients[uid].model,
                    self.server.model,
                    filter_list=self.args.param_filter_list,
                )

                rslt = dict(
                    train_steps=train_steps, train_num=train_num, update_vec=update_vec
                )
                # Step 4
                self.server.collect(uid, rslt)

            # Step 5
            self.server.update(step=step)

            if step % self.args.eval_steps == 0:
                overall_impr = self.evaluate_all_clients(step, load_server_model=True)

                need_stop = self.early_stopper.update(1 - overall_impr)
                if need_stop:
                    logging.info(
                        f"[+] client_{uid} early stops due to worse performance than best result over {self.args.patient} steps"
                    )
                    break

        # distributed model to client and evaluate at the final step
        for uid in self.args.clients:
            self.clients[uid].load_model(
                server_state_dict, filter_list=self.args.param_filter_list
            )
        self.evaluate_all_clients(step)

    def evaluate_all_clients(self, step, load_server_model=False):
        """ Evaluate current global model wtih local evaluation data of all clients

        Args:
            self: 
            step: Current round
            load_server_model: Whether load the current global model for evaluation. If False, evaluate with current local model

        Returns:
            None
        """
        all_relative_impr = []

        server_state_dict = self.server.model.state_dict()

        for uid in self.args.clients:
            if load_server_model:
                self.clients[uid].load_model(
                    server_state_dict, filter_list=self.args.param_filter_list
                )
            eval_rslt = self.clients[uid].eval()
            all_relative_impr.append(eval_rslt["relative_impr"])

            eval_str = "; ".join(
                [f"{metric}: {value}" for metric, value in eval_rslt.items()]
            )
            logging.info(f"client_{uid} step {step}: {eval_str}")

            if "nan" in eval_str:
                logging.info(f"client_{uid} gets nan. Stop training")
                exit()

            self.clients[uid].update_best(
                eval_rslt=eval_rslt,
                state_dict=self.clients[uid].model.state_dict(),
                eval_str=eval_str,
            )

        overall_impr = torch.mean(torch.stack(all_relative_impr))
        logging.info(f"step {step} overall relative_impr {overall_impr}")
        return overall_impr

    def save_all_clients(self):
        """ Save predictions and models for all clients """
        logging.info(f"[+] saving checkpoints and predictions...")
        for uid in self.args.clients:
            self.clients[uid].load_model(self.clients[uid].best_state_dict)
            self.clients[uid].save_prediction(self.args.out_path, dataset="val")
            self.clients[uid].save_prediction(self.args.out_path, dataset="test")
            self.clients[uid].save_best_rslt(self.args.out_path)
            self.clients[uid].save_best_model(self.args.out_path)
            logging.info(f"[-] finish saving predictions for client_{uid}")

    def finetune(self):
        """ Local finetune in the end of federated training

        For each client, finetune the current model for self.args.max_ft_steps.
        Best results are recorded during finetuning and the training will early stop when getting worse performance over self.args.patient steps.

        Args:
            self
        Returns:
            None
        """
        for uid in self.args.clients:
            # finetune from the best FL ckpt
            self.clients[uid].load_model(self.clients[uid].best_state_dict)
            self.early_stopper.clear()

            for epoch in range(self.args.max_ft_steps):
                # local train 1 epoch
                self.clients[uid].finetune(reset_optim=False)
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
            self.clients[uid].save_best_rslt(uid, self.args.out_path)
            self.clients[uid].save_best_model(uid, self.args.out_path)
            logging.info(f"[-] finish saving predictions for client_{uid}")
            del self.clients[uid]

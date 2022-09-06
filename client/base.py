from pathlib import Path
import numpy as np
import os
import logging

import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import VirtualNode

from model import get_model_cls
from metrics import get_metric
from utils import is_name_in_list, split_chunks, merge_chunks


class BaseClient:
    def __init__(self, args, client_config, uid):
        self.args = args
        self.client_config = client_config
        self.uid = uid

        self.task_type = self.client_config["model"]["task"]
        self.out_channels = self.client_config["model"]["out_channels"]
        self.lr = (
            self.client_config["train"]["optimizer"]["lr"]
            if self.args.local_lr is None
            else self.args.local_lr
        )
        if self.args.enable_finetune:
            self.ft_lr = (
                self.client_config["finetune"]["optimizer"]["lr"]
                if self.args.ft_lr is None
                else self.args.ft_lr
            )
        self.major_metric = self.client_config["eval"]["major_metric"]
        self.base_metric = self.client_config["eval"]["base"]

        self.init_model_param()

        self.init_dataset()
        self.init_model()
        self.init_optimizer()
        self.init_metrics()

        self.init_best()

    def init_model_param(self):
        self.model_cls = (
            self.args.model_cls
            if self.args.model_cls
            else self.client_config["model"]["model_cls"]
        )
        self.hidden = (
            self.args.hidden
            if self.args.hidden
            else self.client_config["model"]["hidden"]
        )
        self.max_depth = (
            self.args.max_depth
            if self.args.max_depth
            else self.client_config["model"]["max_depth"]
        )
        self.dropout = (
            self.args.dropout
            if self.args.dropout
            else self.client_config["model"]["dropout"]
        )
        self.pooling = (
            self.args.pooling
            if self.args.pooling
            else self.client_config["model"]["pooling"]
        )

    def init_dataset(self):
        data_path = Path(self.args.data_path) / "CIKM22Competition" / str(self.uid)

        data = {}
        for split in ["train", "val", "test"]:
            data[split] = torch.load(data_path / f"{split}.pt")

        data["train"], data["val"] = self.k_fold_split(data["train"], data["val"])

        self.in_channels = data["train"][0].x.shape[-1]

        data = self.preprocess_data(data)

        dataloader_dict = {}
        dataloader_dict["train"] = DataLoader(
            data["train"], self.args.local_batch_size, shuffle=True
        )
        dataloader_dict["val"] = DataLoader(
            data["val"], self.args.test_batch_size, shuffle=False
        )
        dataloader_dict["test"] = DataLoader(
            data["test"], self.args.test_batch_size, shuffle=False
        )

        self.dataloader_dict = dataloader_dict

    def k_fold_split(self, train_data, val_data):
        # apply k fold cross validation
        if self.args.val_fold != 0:
            train_data_chunks = split_chunks(train_data, self.args.k_fold - 1)
            data_chunks = [val_data] + train_data_chunks
            train_data = merge_chunks(
                [
                    data_chunks[i]
                    for i in range(len(data_chunks))
                    if i != self.args.val_fold
                ]
            )
            val_data = data_chunks[self.args.val_fold]
        return train_data, val_data

    def preprocess_data(self, data):
        if self.pooling == "virtual_node":
            logging.info("[+] Apply virtual node. Preprocessing data")
            transform = VirtualNode()
            for split in ["train", "val", "test"]:
                data[split] = list(map(transform, data[split]))
        return data

    def init_model(self):
        model_cls = get_model_cls(self.model_cls)
        self.model = model_cls(
            self.in_channels,
            self.out_channels,
            hidden=self.hidden,
            max_depth=self.max_depth,
            dropout=self.dropout,
            gnn=self.model_cls,
            pooling=self.pooling,
        )

        if "classification" in self.task_type.lower():
            self.criterion = nn.CrossEntropyLoss()
        elif "regression" in self.task_type.lower():
            self.criterion = nn.MSELoss()

    def init_optimizer(self):
        self.optimizer_cls = eval(f"torch.optim.{self.args.local_optim_cls}")
        self.optimizer = self.optimizer_cls(self.model.parameters(), lr=self.lr)

        if self.args.enable_finetune:
            self.ft_optimizer_cls = eval(f"torch.optim.{self.args.ft_local_optim_cls}")
            self.ft_optimizer = self.ft_optimizer_cls(
                self.model.parameters(), lr=self.ft_lr
            )

    def reset_optimizer(self):
        self.optimizer = self.optimizer_cls(self.model.parameters(), lr=self.lr)

    def reset_ft_optimizer(self):
        self.ft_optimizer = self.ft_optimizer_cls(
            self.model.parameters(), lr=self.ft_lr
        )

    def init_metrics(self):
        self.metric_cals = {}
        assert (
            "relative_impr" in self.client_config["eval"]["metrics"]
        ), f"relative_impr not in metrics of client {self.uid}."

        assert (
            self.major_metric in self.client_config["eval"]["metrics"]
        ), f"Major metric {self.major_metric} not in metrics of client {self.uid}."

        for metric in self.client_config["eval"]["metrics"]:
            # compute relative_impr at the final step of evaluation
            if metric == "relative_impr":
                self.metric_cals[metric] = None
            else:
                self.metric_cals[metric] = get_metric(metric)()

    def init_best(self):
        self.best_rslt, self.best_state_dict, self.best_rslt_str = None, None, None

    def update_best(self, eval_rslt=None, state_dict=None, eval_str=None):
        if self.best_rslt is None or eval_rslt[self.major_metric] <= self.best_rslt:
            self.best_rslt = eval_rslt[self.major_metric]
            self.best_state_dict = state_dict
            self.best_rslt_str = eval_str

    def train(self, reset_optim=True):
        if reset_optim:
            self.reset_optimizer()

        self.model = self.model.cuda()
        self.model.train()

        local_training_steps = 0
        local_training_num = 0

        for epoch in range(self.args.local_epoch):
            for data in self.dataloader_dict["train"]:
                self.optimizer.zero_grad()
                data = data.cuda()
                pred = self.model(data)
                if "classification" in self.task_type.lower():
                    label = data.y.squeeze(-1).long()
                elif "regression" in self.task_type.lower():
                    label = data.y

                if len(label.size()) == 0:
                    label = label.unsqueeze(0)

                loss = self.criterion(pred, label)

                loss.backward()
                self.optimizer.step()

                local_training_steps += 1
                local_training_num += label.shape[0]

        return local_training_num, local_training_steps

    def finetune(self, reset_optim=True):
        if reset_optim:
            self.reset_ft_optimizer()

        self.model = self.model.cuda()
        self.model.train()

        local_training_steps = 0
        local_training_num = 0

        for epoch in range(self.args.ft_local_epoch):
            for data in self.dataloader_dict["train"]:
                self.ft_optimizer.zero_grad()
                data = data.cuda()
                pred = self.model(data)
                if "classification" in self.task_type.lower():
                    label = data.y.squeeze(-1).long()
                elif "regression" in self.task_type.lower():
                    label = data.y

                if len(label.size()) == 0:
                    label = label.unsqueeze(0)

                loss = self.criterion(pred, label)

                loss.backward()
                self.ft_optimizer.step()

                local_training_steps += 1
                local_training_num += label.shape[0]

        return local_training_num, local_training_steps

    @torch.no_grad()
    def eval(self, dataset="val"):
        self.model = self.model.cuda()
        self.model.eval()

        for data in self.dataloader_dict[dataset]:
            data = data.cuda()
            pred = self.model(data)
            if "classification" in self.task_type.lower():
                label = data.y.squeeze(-1).long()
            elif "regression" in self.task_type.lower():
                label = data.y

            if len(label.size()) == 0:
                label = label.unsqueeze(0)

            for metric in self.metric_cals:
                if metric != "relative_impr":
                    self.metric_cals[metric].update(pred, label)

        rslt = {}
        for metric in self.metric_cals:
            if metric != "relative_impr":
                rslt[metric] = self.metric_cals[metric].compute()

        # clear state in metrics
        for metric in self.metric_cals:
            if metric != "relative_impr":
                self.metric_cals[metric].clear()

        if "relative_impr" in self.metric_cals:
            rslt["relative_impr"] = (
                self.base_metric - rslt[self.major_metric]
            ) / self.base_metric

        return rslt

    def load_model(self, state_dict, filter_list=[]):
        if filter_list != []:
            state_dict_filtered = {
                name: param
                for name, param in state_dict.items()
                if not is_name_in_list(name, filter_list)
            }
        else:
            state_dict_filtered = state_dict

        self.model.load_state_dict(state_dict_filtered, strict=False)

    @torch.no_grad()
    def save_prediction(self, path, dataset="test"):
        self.model = self.model.cuda()
        self.model.eval()

        preds = []
        data_indexes = []
        for data in self.dataloader_dict[dataset]:
            data = data.cuda()
            pred = self.model(data)
            if "classification" in self.task_type.lower():
                label = data.y.squeeze(-1).long()
            elif "regression" in self.task_type.lower():
                label = data.y

            if len(label.size()) == 0:
                label = label.unsqueeze(0)

            preds.append(pred.detach().cpu().numpy())
            data_indexes.extend([data[_].data_index.item() for _ in range(len(label))])

        y_inds, y_probs = data_indexes, np.concatenate(preds, axis=0)
        os.makedirs(path, exist_ok=True)

        y_preds = (
            np.argmax(y_probs, axis=-1)
            if "classification" in self.task_type.lower()
            else y_probs
        )

        if len(y_inds) != len(y_preds):
            raise ValueError(
                f"The length of the predictions {len(y_preds)} not equal to the samples {len(y_inds)}."
            )

        # The format of submission file
        with open(os.path.join(path, f"prediction_{dataset}.csv"), "a") as file:
            for y_ind, y_pred in zip(y_inds, y_preds):
                if "classification" in self.task_type.lower():
                    line = [self.uid, y_ind] + [y_pred]
                else:
                    line = [self.uid, y_ind] + list(y_pred)
                file.write(",".join([str(_) for _ in line]) + "\n")

        # save soft predictions
        with open(os.path.join(path, f"prediction_soft_{dataset}.csv"), "a") as file:
            for y_ind, y_pred, y_prob in zip(y_inds, y_preds, y_probs):
                if "classification" in self.task_type.lower():
                    line = [self.uid, y_ind] + list(y_prob)
                else:
                    line = [self.uid, y_ind] + list(y_pred)
                file.write(",".join([str(_) for _ in line]) + "\n")

    def save_best_rslt(self, path):
        with open(os.path.join(path, "eval_rslt.txt"), "a") as file:
            file.write(
                f"client {self.uid} best evaluation result: {self.best_rslt_str} \n"
            )

    def save_best_model(self, path):
        model_path = os.path.join(path, f"model_{self.uid}.pt")
        torch.save(self.best_state_dict, model_path)
        logging.info(f"[+] Save the best model of client {self.uid} at {model_path}")

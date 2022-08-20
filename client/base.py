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
from utils import is_name_in_list


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
        self.major_metric = self.client_config["eval"]["major_metric"]
        self.base_metric = self.client_config["eval"]["base"]

        self.init_dataset()
        self.init_model()
        self.init_optimizer()
        self.init_metrics()

    def init_dataset(self):
        data_path = Path(self.args.data_path) / "CIKM22Competition" / str(self.uid)

        data = {}
        for split in ["train", "val", "test"]:
            data[split] = torch.load(data_path / f"{split}.pt")

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

    def preprocess_data(self, data):
        if self.args.pooling == "virtual_node":
            logging.info("[+] Apply virtual node. Preprocessing data")
            transform = VirtualNode()
            for split in ["train", "val", "test"]:
                data[split] = list(map(transform, data[split]))
        return data

    def init_model(self):
        model_cls = get_model_cls(self.args.model_cls)
        self.model = model_cls(
            self.in_channels,
            self.out_channels,
            hidden=self.args.hidden,
            max_depth=self.args.max_depth,
            dropout=self.args.dropout,
            gnn=self.args.model_cls,
            pooling=self.args.pooling,
        )

        if "classification" in self.task_type.lower():
            self.criterion = nn.CrossEntropyLoss()
        elif "regression" in self.task_type.lower():
            self.criterion = nn.MSELoss()

    def init_optimizer(self):
        self.optimizer_cls = eval(f"torch.optim.{self.args.local_optim_cls}")
        self.optimizer = self.optimizer_cls(self.model.parameters(), lr=self.lr)

    def reset_optimizer(self):
        self.optimizer = self.optimizer_cls(self.model.parameters(), lr=self.lr)

    def init_metrics(self):
        self.metric_cals = {}
        for metric in self.client_config["eval"]["metrics"]:
            # compute relative_impr at the final step of evaluation
            if metric == "relative_impr":
                self.metric_cals[metric] = None
            else:
                self.metric_cals[metric] = get_metric(metric)()

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
    def save_prediction(self, path):
        self.model = self.model.cuda()
        self.model.eval()

        preds = []
        data_indexes = []
        for data in self.dataloader_dict["test"]:
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

        # TODO: more feasible, for now we hard code it for cikmcup
        y_preds = (
            np.argmax(y_probs, axis=-1)
            if "classification" in self.task_type.lower()
            else y_probs
        )

        if len(y_inds) != len(y_preds):
            raise ValueError(
                f"The length of the predictions {len(y_preds)} not equal to the samples {len(y_inds)}."
            )

        with open(os.path.join(path, "prediction.csv"), "a") as file:
            for y_ind, y_pred in zip(y_inds, y_preds):
                if "classification" in self.task_type.lower():
                    line = [self.uid, y_ind] + [y_pred]
                else:
                    line = [self.uid, y_ind] + list(y_pred)
                file.write(",".join([str(_) for _ in line]) + "\n")

    def save_best_rslt(self, uid, eval_str, path):
        with open(os.path.join(path, "eval_rslt.txt"), "a") as file:
            file.write(f"client {uid} best evaluation result: {eval_str} \n")

from pathlib import Path
import numpy as np
import os

import torch
from torch import nn
from torch_geometric.loader import DataLoader

from model import get_model_cls
from metrics import get_metric

class BaseClient:
    def __init__(self, args, client_config, uid):
        self.args = args
        self.client_config = client_config
        self.uid = uid

        self.task_type = self.client_config["model"]["task"]
        self.out_channels = self.client_config["model"]["out_channels"]
        self.lr = self.client_config["train"]["optimizer"]["lr"]
        self.major_metric = self.client_config["eval"]["major_metric"]

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
            
        dataloader_dict = {}
        dataloader_dict['train'] = DataLoader(data['train'], 64, shuffle=True)
        dataloader_dict['val'] = DataLoader(data['val'], 64, shuffle=False)
        dataloader_dict['test'] = DataLoader(data['test'], 64, shuffle=False)

        self.dataloader_dict = dataloader_dict

    def init_model(self):
        model_cls = get_model_cls(self.args.model_cls)
        self.model = model_cls(
            self.in_channels,
            self.out_channels,
            hidden=64,
            max_depth=2,
            dropout=0.2,
            gnn=self.args.model_cls,
            pooling="mean"
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
            self.metric_cals[metric] = get_metric(metric)()

    def train(self, reset_optim=True):
        if reset_optim:
            self.reset_optimizer()

        self.model = self.model.cuda()
        self.model.train()
        for epoch in range(self.args.local_epoch):
            for data in self.dataloader_dict['train']:
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
                self.metric_cals[metric].update(pred, label)
        
        rslt = {}
        for metric in self.metric_cals:
            rslt[metric] = self.metric_cals[metric].compute()
        
        return rslt
            
    def load_model(self, state_dict):
        self.model.load_state_dict(state_dict)

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

        y_inds, y_probs = np.concatenate(data_indexes, axis=0), np.concatenate(preds, axis=0)
        os.makedirs(path, exist_ok=True)

        # TODO: more feasible, for now we hard code it for cikmcup
        y_preds = np.argmax(y_probs, axis=-1) if 'classification' in self.task_type.lower() else y_probs

        if len(y_inds) != len(y_preds):
            raise ValueError(f'The length of the predictions {len(y_preds)} not equal to the samples {len(y_inds)}.')

        with open(os.path.join(path, 'prediction.csv'), 'a') as file:
            for y_ind, y_pred in zip(y_inds,  y_preds):
                if 'classification' in self.task_type.lower():
                    line = [self.uid, y_ind] + [y_pred]
                else:
                    line = [self.uid, y_ind] + list(y_pred)
                file.write(','.join([str(_) for _ in line]) + '\n')
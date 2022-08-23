import logging

import torch
from torch import nn

from client.base import BaseClient
from model import get_model_cls


class HashTensorWrapper:
    def __init__(self, tensor):
        self.tensor = tensor

    def __hash__(self):
        return hash(self.tensor.numpy().tobytes())

    def __eq__(self, other):
        return torch.all(self.tensor == other.tensor)


class RGCNClient(BaseClient):
    def __init__(self, args, client_config, uid):
        super().__init__(args, client_config, uid)
        assert (
            self.model_cls == "rgcn"
        ), f"model_cls must be rgcn for RGCNClient, but get {self.model_cls}"

    def preprocess_data(self, data):
        # add virtual node if pooling as virtual node
        data = super().preprocess_data(data)

        edge_set = set()
        for split in ["train", "val", "test"]:
            for i in data[split]:
                if i.edge_attr is None:
                    break
                edge_set = edge_set | set(
                    [HashTensorWrapper(j) for j in list(i.edge_attr)]
                )

        edge_set = list(set(edge_set))
        self.num_relations = len(edge_set)

        logging.info(f"Client {self.uid} has {self.num_relations} relations.")

        if self.num_relations == 0:
            # normal edge and edge connect with virtual node
            self.num_relations = 2

        edge_type_dict = {i: cnt for cnt, i in enumerate(edge_set)}
        for split in ["train", "val", "test"]:
            for i in data[split]:
                if len(edge_type_dict) == 0:
                    if self.pooling == "virtual_node":
                        # i.edge_type = torch.LongTensor(
                        #     [ 0 for _ in range(i.edge_index.shape[-1]) ]
                        # )
                        virtual_node_index = i.x.shape[0] - 1
                        i.edge_type = torch.LongTensor(
                            [
                                1
                                if i.edge_index[0, j] == virtual_node_index
                                or i.edge_index[1, j] == virtual_node_index
                                else 0
                                for j in range(i.edge_index.shape[-1])
                            ]
                        )
                    else:
                        i.edge_type = torch.LongTensor(
                            [0 for _ in range(i.edge_index.shape[-1])]
                        )
                else:
                    i.edge_type = torch.LongTensor(
                        [edge_type_dict[HashTensorWrapper(j)] for j in i.edge_attr]
                    )

        return data

    def init_model(self):
        model_cls = get_model_cls(self.model_cls)
        self.model = model_cls(
            self.in_channels,
            self.out_channels,
            num_relations=self.num_relations,
            hidden=self.hidden,
            max_depth=self.max_depth,
            dropout=self.dropout,
            pooling=self.pooling,
        )

        if "classification" in self.task_type.lower():
            self.criterion = nn.CrossEntropyLoss()
        elif "regression" in self.task_type.lower():
            self.criterion = nn.MSELoss()

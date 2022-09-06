import torch
from torch.nn import ModuleList
import torch.nn.functional as F
from torch.nn import Linear, Sequential
import torch.nn as nn

from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from torch_geometric.nn.glob import global_add_pool, global_mean_pool, global_max_pool

from model.layers import AtomEncoder, VirtualNodePooling
from .layers import Matcher,MLP
import numpy as np


class RGCN_Net(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_relations,
        hidden=64,
        max_depth=2,
        dropout=0.0,
        num_bases=None,
        root_weight=True
    ):
        super(RGCN_Net, self).__init__()
        self.convs = ModuleList()
        for i in range(max_depth):
            if i == 0:
                self.convs.append(
                    RGCNConv(
                        in_channels,
                        hidden,
                        num_relations=num_relations,
                        num_bases=num_bases,
                        root_weight=root_weight
                    )
                )
            elif (i + 1) == max_depth:
                self.convs.append(
                    RGCNConv(
                        hidden,
                        out_channels,
                        num_relations=num_relations,
                        num_bases=num_bases,
                        root_weight=root_weight
                    )
                )
            else:
                self.convs.append(
                    RGCNConv(
                        hidden, hidden, num_relations=num_relations, num_bases=num_bases,root_weight=root_weight
                    )
                )
        self.dropout = dropout

    def forward(self, data):
        if isinstance(data, Data):
            x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        elif isinstance(data, tuple):
            x, edge_index, edge_type = data
        else:
            raise TypeError("Unsupported data type!")

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if (i + 1) == len(self.convs):
                break
            x = F.relu(F.dropout(x, p=self.dropout, training=self.training))
        return x


class RGCN_Net_Graph(torch.nn.Module):
    r"""GNN model with pre-linear layer, pooling layer
        and output layer for graph classification tasks.

    Arguments:
        in_channels (int): input channels.
        out_channels (int): output channels.
        hidden (int): hidden dim for all modules.
        max_depth (int): number of layers for gnn.
        dropout (float): dropout probability.
        gnn (str): name of gnn type, use ("gcn" or "gin").
        pooling (str): pooling method, use ("add", "mean" or "max").
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_relations,
        hidden=64,
        max_depth=2,
        dropout=0.0,
        pooling="add",
        num_bases=None,
        base_agg="decomposition",
        root_weight=True,
        mode='finetune',
        node_types=None
    ):
        super(RGCN_Net_Graph, self).__init__()
        self.dropout = dropout
        # Embedding (pre) layer
        self.encoder_atom = AtomEncoder(in_channels, hidden)
        self.encoder = Linear(in_channels, hidden)
        self.mode = mode
        if node_types is not None :
            self.node_types = node_types
        if mode == 'pretrain':
            self.matchers = nn.ModuleList()
            self.neg_queue_size = 5
            self.link_dec_dict = {}
            self.neg_queue = {}
            self.link_dec_dict['mol'] = {}
            self.neg_queue['mol'] = {}
            for relation_type in range(num_relations):
                matcher = Matcher(hidden, hidden)
                self.neg_queue['mol'][relation_type] = torch.FloatTensor([]).cuda()
                self.link_dec_dict['mol'][relation_type] = matcher
                self.matchers.append(matcher)
            self.attr_decoder = MLP([hidden, self.node_types], batch_norm=False)
            self.init_emb = nn.Parameter(torch.randn(hidden))
            self.ce = nn.CrossEntropyLoss(reduction = 'none')
            self.neg_samp_num = 4
        # GNN layer

        self.gnn = RGCN_Net(
            in_channels=hidden,
            out_channels=hidden,
            num_relations=num_relations,
            hidden=hidden,
            max_depth=max_depth,
            dropout=dropout,
            num_bases=num_bases,
            root_weight=root_weight
        )

        # Pooling layer
        if pooling == "add":
            self.pooling = global_add_pool
        elif pooling == "mean":
            self.pooling = global_mean_pool
        elif pooling == "max":
            self.pooling = global_max_pool
        elif pooling == "virtual_node":
            self.pooling = VirtualNodePooling()
        else:
            raise ValueError(f"Unsupported pooling type: {pooling}.")

        # Output layer
        self.linear = Sequential(Linear(hidden, hidden), torch.nn.ReLU())
        self.clf = Linear(hidden, out_channels)

    def neg_sample(self, souce_node_list, pos_node_list):
        np.random.shuffle(souce_node_list)
        neg_nodes = []
        keys = {key : True for key in pos_node_list}
        tot       = 0
        for node_id in souce_node_list:
            if node_id not in keys:
                neg_nodes += [node_id]
                tot += 1
            if tot == self.neg_samp_num:
                break
        return neg_nodes

    def feat_loss(self, reps, node_cls):
        logits = self.attr_decoder(reps)
        loss = self.ce(logits,node_cls)
        return loss.mean()

    def link_loss(self, node_emb, rem_edge_list, ori_edge_list, node_dict, target_type, use_queue = True, update_queue = False):
        losses = 0
        train_num=0
        ress   = []
        for source_type in rem_edge_list:
            if source_type not in self.link_dec_dict:
                continue
            for relation_type in rem_edge_list[source_type]:
                if relation_type not in self.link_dec_dict[source_type]:
                    continue
                rem_edges = rem_edge_list[source_type][relation_type]
                if len(rem_edges) <1:
                    continue
                ori_edges = ori_edge_list[source_type][relation_type]
                matcher = self.link_dec_dict[source_type][relation_type]
                target_ids, positive_source_ids = rem_edges[:,0].reshape(-1, 1), rem_edges[:,1].reshape(-1, 1)
                n_nodes = len(target_ids)
                source_node_ids = np.unique(ori_edges[:, 1])

                negative_source_ids = [self.neg_sample(source_node_ids, \
                    ori_edges[ori_edges[:, 0] == t_id][:, 1].tolist()) for t_id in target_ids]
                sn = min([len(neg_ids) for neg_ids in negative_source_ids])

                negative_source_ids = [neg_ids[:sn] for neg_ids in negative_source_ids]

                source_ids = torch.LongTensor(np.concatenate((positive_source_ids, negative_source_ids), axis=-1) + node_dict[source_type][0])
                emb = node_emb[source_ids]
                
                if use_queue and len(self.neg_queue[source_type][relation_type]) // n_nodes > 0:
                    tmp = self.neg_queue[source_type][relation_type]
                    stx = len(tmp) // n_nodes
                    tmp = tmp[: stx * n_nodes].reshape(n_nodes, stx, -1)
                    rep_size = sn + 1 + stx
                    source_emb = torch.cat([emb, tmp], dim=1)
                    source_emb = source_emb.reshape(n_nodes * rep_size, -1)
                else:
                    rep_size = sn + 1
                    source_emb = emb.reshape(source_ids.shape[0] * rep_size, -1)
                    
                target_ids = target_ids.repeat(rep_size, 1) + node_dict[target_type][0]
                target_emb = node_emb[target_ids.reshape(-1)]
                res = matcher.forward(target_emb, source_emb)
                res = res.reshape(n_nodes, rep_size)
                train_num += res.shape[0]
                ress += [res.detach()]
                losses += F.log_softmax(res, dim=-1)[:,0].mean()
                if update_queue and 'L1' not in relation_type and 'L2' not in relation_type:
                    tmp = self.neg_queue[source_type][relation_type]
                    self.neg_queue[source_type][relation_type] = \
                        torch.cat([node_emb[source_node_ids].detach(), tmp], dim=0)[:int(self.neg_queue_size * n_nodes)]
        return -losses / len(ress) if len(ress)!=0 else 0, ress, train_num


    def forward(self, data,start_idx=None,end_idx=None):
        if isinstance(data, Batch):
            x, edge_index, edge_type, batch = (
                data.x,
                data.edge_index,
                data.edge_type,
                data.batch,
            )
        elif isinstance(data, tuple):
            x, edge_index, edge_type, batch = data
        else:
            raise TypeError("Unsupported data type!")

        if x.dtype == torch.int64:
            x = self.encoder_atom(x)
        else:
            x = self.encoder(x)
        if self.mode == 'pretrain':
            x[start_idx : end_idx] = self.init_emb

        x = self.gnn((x, edge_index, edge_type))
        if self.mode == 'pretrain':
            return x
        if isinstance(self.pooling,VirtualNodePooling):
            batch_diff = batch - batch[(torch.arange(len(batch)) + 1) % len(batch)]
            last_indices = batch_diff != 0
            x = x[last_indices]
        else:
            x = self.pooling(x, batch)
        x = self.linear(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.clf(x)
        return x

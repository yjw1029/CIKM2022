import torch
from torch.nn import ModuleList
import torch.nn.functional as F
from torch.nn import Linear, Sequential

from torch_geometric.nn import GINEConv
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from torch_geometric.nn.glob import global_add_pool, global_mean_pool, global_max_pool

from model.layers import AtomEncoder, VirtualNodePooling, MLP, EdgeEncoder


class GINE_Net(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden=64,
        max_depth=2,
        dropout=0.0,
    ):
        super(GINE_Net, self).__init__()
        self.convs = ModuleList()
        for i in range(max_depth):
            if i == 0:
                self.convs.append(
                    GINEConv(
                        MLP([in_channels, hidden, hidden], batch_norm=True),
                        edge_dim=hidden,
                    )
                )
            elif (i + 1) == max_depth:
                self.convs.append(
                    GINEConv(
                        MLP([hidden, hidden, out_channels], batch_norm=True),
                        edge_dim=hidden,
                    )
                )
            else:
                self.convs.append(
                    GINEConv(
                        MLP([hidden, hidden, hidden], batch_norm=True), edge_dim=hidden
                    )
                )
        self.dropout = dropout

    def forward(self, data):
        if isinstance(data, Data):
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        elif isinstance(data, tuple):
            x, edge_index, edge_attr = data
        else:
            raise TypeError("Unsupported data type!")

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            if (i + 1) == len(self.convs):
                break
            x = F.relu(F.dropout(x, p=self.dropout, training=self.training))
        return x


class GINE_Net_Graph(torch.nn.Module):
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
        **kwargs,
    ):
        super(GINE_Net_Graph, self).__init__()
        self.dropout = dropout
        # Embedding (pre) layer
        self.encoder_atom = AtomEncoder(in_channels, hidden)
        self.encoder = Linear(in_channels, hidden)

        # Edge Embedding (pre) layer
        self.encoder_egde = EdgeEncoder(num_relations, hidden)

        # GNN layer

        self.gnn = GINE_Net(
            in_channels=hidden,
            out_channels=hidden,
            num_relations=num_relations,
            hidden=hidden,
            max_depth=max_depth,
            dropout=dropout,
            num_bases=num_bases,
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

    def forward(self, data):
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

        edge_attr = self.encoder_egde(edge_type)

        x = self.gnn((x, edge_index, edge_attr))
        x = self.pooling(x, batch)
        x = self.linear(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.clf(x)
        return x

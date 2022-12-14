import torch
import torch.nn as nn
from torch.nn import ModuleList
import torch.nn.functional as F
from torch.nn import Linear, Sequential
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from torch_geometric.nn.glob import global_add_pool, global_mean_pool, \
    global_max_pool
from model.layers import AtomEncoder, VirtualNodePooling
from torch.nn import Parameter
from torch.nn import Parameter as Param
from torch_scatter import scatter
from torch_geometric.nn.conv import MessagePassing
from typing import Optional, Tuple, Union
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor, masked_select_nnz, matmul
from .layers import MLP
import copy

try:
    from pyg_lib.ops import segment_matmul  # noqa
    _WITH_PYG_LIB = True
except ImportError:
    _WITH_PYG_LIB = False

    def segment_matmul(inputs: Tensor, ptr: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError

@torch.jit._overload
def masked_edge_index(edge_index, edge_mask):
    # type: (Tensor, Tensor) -> Tensor
    pass


@torch.jit._overload
def masked_edge_index(edge_index, edge_mask):
    # type: (SparseTensor, Tensor) -> SparseTensor
    pass


def masked_edge_index(edge_index, edge_mask):
    if isinstance(edge_index, Tensor):
        return edge_index[:, edge_mask]
    else:
        return masked_select_nnz(edge_index, edge_mask, layout='coo')
        
class RGINConv(MessagePassing):
    '''
    RGINConv is based on GIN and considers edge attributes.

    Arguments:
        in_channels: dimension of input.
        out_channels: dimension of output.
        hidden: dimension of hidden units, default=64.
        num_relations: the number of edge types.
        num_bases: the number of base matrices
        num_blocks: Optional[int] = None,
        aggr: the strategy of aggregation
        root_weight: whether the model adds transformed root node features to the output
        bias: whether the model uses the bias
        base_agg: str = the strategy of combining base matrices
        comp: the weights of combining base matrices
        MLP: the base matrices or matrices for different edge types
    '''
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        hidden:int,
        num_relations: int,
        num_bases: Optional[int] = None,
        num_blocks: Optional[int] = None,
        aggr: str = 'add',
        root_weight: bool = True,
        is_sorted: bool = False,
        bias: bool = True,
        base_agg: str = 'decomposition', #(1)decomposition (2) moe
        **kwargs,
    ):
        kwargs.setdefault('aggr', aggr)
        super().__init__(node_dim=0, **kwargs)
        self._WITH_PYG_LIB = torch.cuda.is_available() and _WITH_PYG_LIB

        if num_bases is not None and num_blocks is not None:
            raise ValueError('Can not apply both basis-decomposition and '
                             'block-diagonal-decomposition at the same time.')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_blocks = num_blocks
        self.is_sorted = is_sorted
        self.base_agg = base_agg

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.in_channels_l = in_channels[0]

        if num_bases is not None:
            if self.base_agg == 'decomposition':
                if root_weight:
                    self.MLP = MLP([self.in_channels_l, hidden, out_channels], batch_norm=True,num_relations=num_relations+1,num_bases=num_bases) #contain root weight
                    self.root = True
                else:
                    self.MLP = MLP([self.in_channels_l, hidden, out_channels], batch_norm=True,num_relations=num_relations,num_bases=num_bases)
            elif self.base_agg == 'moe':
                self.MLP = [MLP([self.in_channels_l, hidden, out_channels], batch_norm=True).cuda() for i in range(num_bases)]
                if root_weight:
                    self.comp = Parameter(torch.Tensor(num_relations+1, num_bases))
                    self.root = True
                else:
                    self.comp = Parameter(torch.Tensor(num_relations, num_bases))
        else:
            self.MLP = [MLP([self.in_channels_l, hidden, out_channels], batch_norm=True).cuda() for i in range(num_relations)]
            self.register_parameter('comp', None)

        if root_weight:
            if num_bases is None:
                self.root = MLP([self.in_channels_l, hidden, out_channels], batch_norm=True).cuda()
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        '''
        reset the model parameters
        '''
        if self.num_bases is not None:
            if self.base_agg == 'decomposition':
                self.MLP.reset_parameters()
            elif self.base_agg == 'moe':
                for i in self.MLP:
                    i.reset_parameters()
        else:
            for i in self.MLP:
                i.reset_parameters()
            self.root.reset_parameters()
        nn.init.zeros_(self.bias)

    def weighted_average(self,x,comp):
        '''
        Args:
            x: data
            comp: the weight of combining base models
        Return:
            y: the outputs of MoE

        aggregate the results by MoE (Mixture of Experts)
        '''
        y = None
        for i,mlp in enumerate(self.MLP):
            if y is None:
                y = mlp(x)*comp[i]
            else:
                y += mlp(x)*comp[i]
        return y

    def forward(self, x: Union[OptTensor, Tuple[OptTensor, Tensor]],
                edge_index: Adj, edge_type: OptTensor = None):
        '''
        Args:
            x: node feature
            edge_index: the edges of graphs
            edge_type: the type for each edge 

        Return:
            out: the output of RGINConv
        '''
        x_l: OptTensor = None
        if isinstance(x, tuple):
            x_l = x[0]
        else:
            x_l = x
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)

        x_r: Tensor = x_l
        if isinstance(x, tuple):
            x_r = x[1]

        size = (x_l.size(0), x_r.size(0))

        if isinstance(edge_index, SparseTensor):
            edge_type = edge_index.storage.value()
        assert edge_type is not None

        # propagate_type: (x: Tensor, edge_type_ptr: OptTensor)
        out = torch.zeros(x_r.size(0), self.out_channels, device=x_r.device)

        root = self.root
        MLP = self.MLP

        for i in range(self.num_relations):
            tmp = masked_edge_index(edge_index, edge_type == i)

            if x_l.dtype == torch.long:
                x_l = x_l.float()
            h = self.propagate(tmp, x=x_l, edge_type_ptr=None,
                                size=size)
            if self.num_bases is None:
                out = out + MLP[i](h)
            else:
                if self.base_agg == 'decomposition':
                    out = out + MLP(h,i)
                elif self.base_agg == 'moe':
                    out = out + self.weighted_average(h,self.comp[i])
        
        if root is not None:
            if self.num_bases is None:
                out += root(x_r.float()) if x_r.dtype == torch.long else root(x_r)
            else:
                if self.base_agg == 'decomposition':
                    out += MLP(x_r.float(),self.num_relations) if x_r.dtype == torch.long else MLP(x_r,self.num_relations)
                elif self.base_agg == 'moe':
                    out = out + ( self.weighted_average(x_r.float(),self.comp[self.num_relations]) if x_r.dtype == torch.long else self.weighted_average(x_r,self.comp[self.num_relations]) )

        if self.bias is not None:
            out += self.bias

        return out


    def message(self, x_j: Tensor, edge_type_ptr: OptTensor) -> Tensor:
        '''
        pass messages
        '''
        if edge_type_ptr is not None:
            return segment_matmul(x_j, edge_type_ptr, self.weight)

        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        '''
        pass messages and aggregate messages
        '''
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_relations={self.num_relations})')

class RGIN_Net(torch.nn.Module):
    '''

    Arguments:
        in_channels: dimension of input.
        out_channels: dimension of output.
        hidden: dimension of hidden units, default=64.
        num_relations: the number of edge types.
        num_bases: the number of base matrices
        base_agg: str = the strategy of combining base matrices
        max_depth: layers of GNN
        dropout (float): dropout ratio
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_relations,
                 hidden=64,
                 max_depth=2,
                 dropout=.0,
                 num_bases=None,
                 base_agg='decomposition'):
        super(RGIN_Net, self).__init__()
        self.convs = ModuleList()
        for i in range(max_depth):
            if i == 0:
                self.convs.append(
                    RGINConv(in_channels, hidden,hidden, num_relations=num_relations,num_bases=num_bases,base_agg=base_agg))
            elif (i + 1) == max_depth:
                self.convs.append(
                    RGINConv(hidden,hidden, out_channels, num_relations=num_relations,num_bases=num_bases,base_agg=base_agg))
            else:
                self.convs.append(
                    RGINConv(hidden,hidden, hidden, num_relations=num_relations,num_bases=num_bases,base_agg=base_agg))
        self.dropout = dropout

    def forward(self, data):
        if isinstance(data, Data):
            x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        elif isinstance(data, tuple):
            x, edge_index, edge_type = data
        else:
            raise TypeError('Unsupported data type!')

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if (i + 1) == len(self.convs):
                break
            x = F.relu(F.dropout(x, p=self.dropout, training=self.training))
        return x

    def reset_parameters(self):
        for m in self.convs:
            m.reset_parameters()


class RGIN_Net_Graph(torch.nn.Module):
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
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_relations,
                 hidden=64,
                 max_depth=2,
                 dropout=.0,
                 pooling='add',
                 num_bases=None,
                 base_agg='decomposition'):
        super(RGIN_Net_Graph, self).__init__()
        self.dropout = dropout
        # Embedding (pre) layer
        self.encoder_atom = AtomEncoder(in_channels, hidden)
        self.encoder = Linear(in_channels, hidden)
        # GNN layer
        
        self.gnn = RGIN_Net(in_channels=hidden,
                           out_channels=hidden,
                           num_relations=num_relations,
                           hidden=hidden,
                           max_depth=max_depth,
                           dropout=dropout,
                           num_bases=num_bases,
                           base_agg=base_agg)
        
        # Pooling layer
        if pooling == 'add':
            self.pooling = global_add_pool
        elif pooling == 'mean':
            self.pooling = global_mean_pool
        elif pooling == 'max':
            self.pooling = global_max_pool
        elif pooling == "virtual_node":
            self.pooling = VirtualNodePooling()
        else:
            raise ValueError(f'Unsupported pooling type: {pooling}.')

        # Output layer
        self.linear = Sequential(Linear(hidden, hidden), torch.nn.ReLU())
        self.clf = Linear(hidden, out_channels)

    def forward(self, data):
        '''
        the compute function of GNN model
        '''
        if isinstance(data, Batch):
            x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_type, data.batch
        elif isinstance(data, tuple):
            x, edge_index, edge_type, batch = data
        else:
            raise TypeError('Unsupported data type!')

        if x.dtype == torch.int64:
            x = self.encoder_atom(x)
        else:
            x = self.encoder(x)

        x = self.gnn((x, edge_index, edge_type))
        x = self.pooling(x, batch)
        x = self.linear(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.clf(x)
        return x

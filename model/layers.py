import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch.nn import BatchNorm1d, Identity
import torch.nn.init as init
from torch.nn import Parameter
import torch.nn as nn
from torch import Tensor
import math

EMD_DIM = 200


class AtomEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden):
        super(AtomEncoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        for i in range(in_channels):
            emb = torch.nn.Embedding(EMD_DIM, hidden)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])
        return x_embedding

class VirtualNodePooling(torch.nn.Module):
    def __init__(self):
        super(VirtualNodePooling, self).__init__()

    def forward(self, x, batch):
        batch_diff = batch - batch[(torch.arange(len(batch)) + 1) % len(batch)]
        last_indices = batch_diff != 0
        return x[last_indices]

class MLP(torch.nn.Module):
    """
    Multilayer Perceptron
    """
    def __init__(self,
                 channel_list,
                 dropout=0.,
                 batch_norm=True,
                 relu_first=False,
                 num_bases=None,
                 num_relations=None):
        super().__init__()
        assert len(channel_list) >= 2
        self.channel_list = channel_list
        self.dropout = dropout
        self.relu_first = relu_first
        self.num_bases = num_bases
        self.num_relations = num_relations

        self.linears = ModuleList()
        self.norms = ModuleList()
        for in_channel, out_channel in zip(channel_list[:-1],
                                           channel_list[1:]):
            if num_bases is None:
                self.linears.append(Linear(in_channel, out_channel))
            else:
                self.linears.append(Linear_w_base(in_channel, out_channel, num_bases=num_bases, num_relations=num_relations))
            self.norms.append(
                BatchNorm1d(out_channel) if batch_norm else Identity())

    def reset_parameters(self):
        for lin in self.linears:
            lin.reset_parameters()

    def forward(self, x,relation=None):
        if relation is None:
            x = self.linears[0](x)
        else:
            x = self.linears[0](x,relation)
        for layer, norm in zip(self.linears[1:], self.norms[:-1]):
            if self.relu_first:
                x = F.relu(x)
            x = norm(x)
            if not self.relu_first:
                x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if relation is None:
                x = layer.forward(x)
            else:
                x = layer.forward(x,relation)
        return x


class Linear_w_base(torch.nn.Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None,num_bases=None,num_relations=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear_w_base, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_bases = num_bases
        self.num_relations = num_relations
        if num_bases is None:
            self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        else:
            self.weight = Parameter(torch.empty((num_bases,out_features, in_features)))
            self.comp = Parameter(torch.Tensor(num_relations, num_bases))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
        if self.num_bases is not None:
            init.uniform_(self.comp)



    def forward(self, input: Tensor,relation=None) -> Tensor:
        if self.num_bases is None:
            return F.linear(input, self.weight, self.bias)
        else:
            weight = (self.comp[relation].unsqueeze(0)@ self.weight.view(self.num_bases,-1)).view(self.out_features,self.in_features)
            return F.linear(input,weight,self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class Matcher(nn.Module):
    '''
        Matching between a pair of nodes to conduct link prediction.
        Use multi-head attention as matching model.
    '''
    
    def __init__(self, n_hid, n_out, temperature = 0.1):
        super(Matcher, self).__init__()
        self.n_hid          = n_hid
        self.linear    = nn.Linear(n_hid,  n_out)
        self.sqrt_hd     = math.sqrt(n_out)
        self.drop        = nn.Dropout(0.2)
        self.cosine      = nn.CosineSimilarity(dim=1)
        self.cache       = None
        self.temperature = temperature
    def forward(self, x, ty, use_norm = True):
        tx = self.drop(self.linear(x))
        if use_norm:
            return self.cosine(tx, ty) / self.temperature
        else:
            return (tx * ty).sum(dim=-1) / self.sqrt_hd
    def __repr__(self):
        return '{}(n_hid={})'.format(
            self.__class__.__name__, self.n_hid)
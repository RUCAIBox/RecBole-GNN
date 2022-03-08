import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class GCNConv(MessagePassing):
    def __init__(self, dim):
        super(GCNConv, self).__init__(aggr='add')
        self.dim = dim

    def forward(self, x, edge_index, edge_weight):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.dim)


class SRGNNConv(MessagePassing):
    def __init__(self, dim):
        super(SRGNNConv, self).__init__(aggr='mean')

        self.lin = torch.nn.Linear(dim, dim)
        self.b = nn.Parameter(torch.Tensor(dim))

    def forward(self, x, edge_index):
        x = self.lin(x)
        return self.propagate(edge_index, x=x) + self.b


class SRGNNCell(nn.Module):
    def __init__(self, dim):
        super(SRGNNCell, self).__init__()

        self.incomming_conv = SRGNNConv(dim)
        self.outcomming_conv = SRGNNConv(dim)

        self.lin_ih = nn.Linear(2 * dim, 3 * dim)
        self.lin_hh = nn.Linear(dim, 3 * dim)

    def forward(self, hidden, edge_index):
        input_in = self.incomming_conv(hidden, edge_index)
        reversed_edge_index = torch.flip(edge_index, dims=[0])
        input_out = self.outcomming_conv(hidden, reversed_edge_index)
        inputs = torch.cat([input_in, input_out], dim=-1)

        gi = self.lin_ih(inputs)
        gh = self.lin_hh(hidden)
        i_r, i_i, i_n = gi.chunk(3, -1)
        h_r, h_i, h_n = gh.chunk(3, -1)
        reset_gate = torch.sigmoid(i_r + h_r)
        input_gate = torch.sigmoid(i_i + h_i)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        hy = (1 - input_gate) * hidden + input_gate * new_gate
        return hy

# modified from https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py

import math

import torch
import torch.nn as nn

from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.bn = nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.bmm(adj, x)
        result = torch.mm(support.view(-1, self.in_features), self.weight)
        output = result.view(-1, adj.data.shape[1], self.out_features)
        if self.bias is not None:
            output = output + self.bias
        output = output.transpose(1, 2).contiguous()
        output = self.bn(output)
        output = output.transpose(1, 2)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

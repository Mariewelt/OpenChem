# TODO: encoding of molecular graph into vector

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.encoders.openchem_encoder import OpenChemEncoder
from utils.utils import check_params
from layers.gcn import GraphConvolution


class GraphCNNEncoder(OpenChemEncoder):
    def __init__(self, params, use_cuda):
        super(GraphCNNEncoder, self).__init__(params, use_cuda)
        check_params(params, self.get_required_params(),
                     self.get_optional_params())
        self.n_layers = params['n_layers']
        self.hidden_size = params['hidden_size']
        self.dropout = params['dropout']
        assert len(self.hidden_size) == self.n_layers
        #self.hidden_size += [self.encoder_dim]
        self.hidden_size = [self.input_size] + self.hidden_size
        self.graph_convolutions = nn.ModuleList()
        self.dense = nn.Linear(in_features=self.hidden_size[-1],
                                      out_features=self.encoder_dim)
        for i in range(1, self.n_layers+1):
            self.graph_convolutions.append(GraphConvolution(self.
                                                            hidden_size[i-1],
                                                            self.
                                                            hidden_size[i]))

    @staticmethod
    def get_optional_params():
        return {
            'dropout': float
        }

    @staticmethod
    def get_required_params():
        return {
            'n_layers': int,
            'hidden_size': list,
        }

    def forward(self, inp):
        x = inp[0]
        adj = inp[1]
        for i in range(self.n_layers):
            x = self.graph_convolutions[i](x, adj)
            #if i < self.n_layers - 1:
            #    x = F.dropout(x, self.dropout)
            x = F.relu(x)
            x = torch.bmm(adj, x)
        x = F.tanh(self.dense(x))
        x = F.tanh(x.sum(dim=1))
        return x

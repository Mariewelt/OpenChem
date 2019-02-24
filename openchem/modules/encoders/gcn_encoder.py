import torch
import torch.nn as nn

from openchem.modules.encoders.openchem_encoder import OpenChemEncoder
from openchem.utils.utils import check_params
from openchem.layers.gcn import GraphConvolution


class GraphCNNEncoder(OpenChemEncoder):
    def __init__(self, params, use_cuda):
        super(GraphCNNEncoder, self).__init__(params, use_cuda)
        check_params(params, self.get_required_params(),
                     self.get_optional_params())
        self.n_layers = params['n_layers']
        self.hidden_size = params['hidden_size']
        if 'dropout' in params.keys():
            self.dropout = params['dropout']
        else:
            self.dropout = 0
        assert len(self.hidden_size) == self.n_layers
        self.hidden_size = [self.input_size] + self.hidden_size
        self.graph_convolutions = nn.ModuleList()
        self.dropout_layer = nn.Dropout(p=self.dropout)
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
            x = torch.tanh(x)
            n = adj.size(1)
            d = x.size()[-1]
            adj_new = adj.unsqueeze(3)
            adj_new = adj_new.expand(-1, n, n, d)
            x_new = x.repeat(1, n, 1).view(-1, n, n, d)
            res = x_new*adj_new
            x = res.max(dim=2)[0]
        x = torch.tanh(self.dense(x))
        x = torch.tanh(x.sum(dim=1))
        return x

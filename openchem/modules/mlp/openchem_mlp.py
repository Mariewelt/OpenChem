import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from openchem.utils.utils import check_params


class OpenChemMLP(nn.Module):
    """Base class for MLP module"""
    def __init__(self, params):
        super(OpenChemMLP, self).__init__()
        check_params(params, self.get_required_params(), self.get_optional_params())
        self.params = params
        self.hidden_size = self.params['hidden_size']
        self.input_size = [self.params['input_size']] + self.hidden_size[:-1]
        self.n_layers = self.params['n_layers']
        self.activation = self.params['activation']
        if type(self.activation) is list:
            assert len(self.activation) == self.n_layers
        else:
            self.activation = [self.activation] * self.n_layers
        if 'dropout' in self.params.keys():
            self.dropout = self.params['dropout']
        else:
            self.dropout = 0
        self.layers = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.dropouts = nn.ModuleList([])
        for i in range(self.n_layers - 1):
            self.dropouts.append(nn.Dropout(self.dropout))
            self.bn.append(nn.BatchNorm1d(self.hidden_size[i]))
            self.layers.append(nn.Linear(in_features=self.input_size[i], out_features=self.hidden_size[i]))
        i = self.n_layers - 1
        self.dropouts.append(nn.Dropout(self.dropout))
        self.layers.append(nn.Linear(in_features=self.input_size[i], out_features=self.hidden_size[i]))

    @staticmethod
    def get_required_params():
        return {
            'input_size': int,
            'n_layers': int,
            'hidden_size': list,
            'activation': None,
        }

    @staticmethod
    def get_optional_params():
        return {'dropout': float}

    def forward(self, inp):
        output = inp
        for i in range(self.n_layers - 1):
            output = self.dropouts[i](output)
            output = self.layers[i](output)
            output = self.bn[i](output)
            output = self.activation[i](output)
        output = self.dropouts[-1](output)
        output = self.layers[-1](output)
        output = self.activation[-1](output)
        return output


class OpenChemMLPSimple(nn.Module):
    """Base class for MLP module"""
    def __init__(self, params):
        super(OpenChemMLPSimple, self).__init__()
        check_params(params, self.get_required_params(), self.get_optional_params())
        self.params = params
        self.hidden_size = self.params['hidden_size']
        self.input_size = [self.params['input_size']] + self.hidden_size[:-1]
        self.n_layers = self.params['n_layers']
        self.activation = self.params['activation']
        if type(self.activation) is list:
            assert len(self.activation) == self.n_layers
        else:
            self.activation = [self.activation] * self.n_layers

        self.layers = nn.ModuleList([])
        for i in range(self.n_layers):
            self.layers.append(nn.Linear(in_features=self.input_size[i], out_features=self.hidden_size[i]))

        if "init" in self.params.keys():
            if self.params["init"] == "xavier_uniform":
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            else:
                raise NotImplementedError("Only xavier_uniform "
                                          "initialization is "
                                          "supported now in OpenChemMLPSimple")

    @staticmethod
    def get_required_params():
        return {
            'input_size': int,
            'n_layers': int,
            'hidden_size': list,
            'activation': None,
        }

    @staticmethod
    def get_optional_params():
        return {'init': str}

    def forward(self, inp):
        output = inp
        for i in range(self.n_layers):
            output = self.layers[i](output)
            output = self.activation[i](output)
        return output

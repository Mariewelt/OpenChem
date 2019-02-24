import torch
import numpy as np
from torch import nn
from openchem.modules.encoders.openchem_encoder import OpenChemEncoder

from openchem.utils.utils import check_params


class RNNEncoder(OpenChemEncoder):
    def __init__(self, params, use_cuda):
        super(RNNEncoder, self).__init__(params, use_cuda)
        check_params(params, self.get_required_params(),
                     self.get_optional_params())
        self.layer = self.params['layer']
        layers = ['LSTM', 'GRU', 'RNN']
        if self.layer not in ['LSTM', 'GRU', 'RNN']:
            raise ValueError(self.layer + ' is invalid value for argument'
                                          ' \'layer\'. Choose one from :'
                             + ', '.join(layers))

        self.input_size = self.params['input_size']
        self.encoder_dim = self.params['encoder_dim']
        self.n_layers = self.params['n_layers']
        if self.n_layers > 1:
            self.dropout = self.params['dropout']
        else:
            UserWarning('dropout can be non zero only when n_layers > 1. '
                        'Parameter dropout set to 0.')
            self.dropout = 0
        self.bidirectional = self.params['is_bidirectional']
        if self.bidirectional:
            self.n_directions = 2
        else:
            self.n_directions = 1
        if self.layer == 'LSTM':
            self.rnn = nn.LSTM(self.input_size, self.encoder_dim,
                               self.n_layers,
                               bidirectional=self.bidirectional,
                               dropout=self.dropout,
                               bias=True)
        elif self.layer == 'GRU':
            self.rnn = nn.GRU(self.input_size, self.encoder_dim,
                              self.n_layers,
                              bidirectional=self.bidirectional,
                              dropout=self.dropout,
                              bias=True)
        else:
            self.layer = nn.RNN(self.input_size, self.encoder_dim,
                                self.n_layers,
                                bidirectional=self.bidirectional,
                                dropout=self.dropout,
                                bias=True)

    @staticmethod
    def get_required_params():
        return {
            'input_size': int,
            'encoder_dim': int,
        }

    @staticmethod
    def get_optional_params():
        return{
            'layer': str,
            'n_layers': int,
            'dropout': float,
            'is_bidirectional': bool
        }

    def forward(self, inp, previous_hidden=None):
        """
        inp: shape batch_size, seq_len, input_size
        previous_hidden: if given shape n_layers * num_directions,
        batch_size, embedding_dim.
               Initialized automatically if None
        return: embedded
        """
        input_tensor = inp[0]
        input_tensor = input_tensor.permute(1, 0, 2)
        input_length = inp[1]
        batch_size = input_tensor.size()[1]
        if previous_hidden is None:
            previous_hidden = self.init_hidden(batch_size)
            if self.layer == 'LSTM':
                cell = self.init_cell(batch_size)
                previous_hidden = (previous_hidden, cell)
        output, _ = self.rnn(input_tensor, previous_hidden)
        index_tensor = input_length.cpu().numpy() - 1
        index_tensor = np.array([index_tensor]).astype('int')
        index_tensor = np.repeat(np.array([index_tensor]), 
                                 repeats=output.size()[2], 
                                 axis=0)
        index_tensor = index_tensor.swapaxes(0, 1)
        index_tensor = index_tensor.swapaxes(1, 2)
        index_tensor = torch.LongTensor(index_tensor).cuda()
        embedded = torch.gather(output, dim=0, index=index_tensor).squeeze(0)
        return embedded, previous_hidden

    def init_hidden(self, batch_size):
        if self.use_cuda:
            return torch.tensor(torch.zeros(self.n_layers * self.n_directions,
                                            batch_size,
                                            self.encoder_dim),
                                requires_grad=True).cuda()
        else:
            return torch.tensor(torch.zeros(self.n_layers * self.n_directions,
                                            batch_size,
                                            self.encoder_dim),
                                requires_grad=True)

    def init_cell(self, batch_size):
        if self.use_cuda:
            return torch.tensor(torch.zeros(self.n_layers * self.n_directions,
                                            batch_size,
                                            self.encoder_dim),
                                requires_grad=True).cuda()
        else:
            return torch.tensor(torch.zeros(self.n_layers * self.n_directions,
                                            batch_size,
                                            self.encoder_dim),
                                requires_grad=True)


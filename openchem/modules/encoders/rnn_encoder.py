import torch
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
                               dropout=self.dropout)
        elif self.layer == 'GRU':
            self.rnn = nn.GRU(self.input_size, self.encoder_dim,
                              self.n_layers,
                              bidirectional=self.bidirectional,
                              dropout=self.dropout)
        else:
            self.layer = nn.RNN(self.input_size, self.encoder_dim,
                                self.n_layers,
                                bidirectional=self.bidirectional,
                                dropout=self.dropout)

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
        inp = inp.permute(1, 0, 2)
        batch_size = inp.size()[1]
        if previous_hidden is None:
            previous_hidden = self.init_hidden(batch_size)
            if self.layer == 'LSTM':
                cell = self.init_cell(batch_size)
                previous_hidden = (previous_hidden, cell)
        output, _ = self.rnn(inp, previous_hidden)
        embedded = output[-1, :, :].squeeze(0)
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


import torch
import numpy as np
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from openchem.modules.encoders.openchem_encoder import OpenChemEncoder

from openchem.utils.utils import check_params


class RNNEncoder(OpenChemEncoder):
    def __init__(self, params, use_cuda):
        super(RNNEncoder, self).__init__(params, use_cuda)
        check_params(params, self.get_required_params(), self.get_optional_params())
        self.layer = self.params['layer']
        layers = ['LSTM', 'GRU', 'RNN']
        if self.layer not in ['LSTM', 'GRU', 'RNN']:
            raise ValueError(self.layer + ' is invalid value for argument'
                             ' \'layer\'. Choose one from :' + ', '.join(layers))

        self.input_size = self.params['input_size']
        self.encoder_dim = self.params['encoder_dim']
        self.n_layers = self.params['n_layers']
        if self.n_layers > 1:
            self.dropout = self.params['dropout']
        else:
            UserWarning('dropout can be non zero only when n_layers > 1. ' 'Parameter dropout set to 0.')
            self.dropout = 0
        self.bidirectional = self.params['is_bidirectional']
        if self.bidirectional:
            self.n_directions = 2
        else:
            self.n_directions = 1
        if self.layer == 'LSTM':
            self.rnn = nn.LSTM(self.input_size,
                               self.encoder_dim,
                               self.n_layers,
                               bidirectional=self.bidirectional,
                               dropout=self.dropout,
                               batch_first=True)
        elif self.layer == 'GRU':
            self.rnn = nn.GRU(self.input_size,
                              self.encoder_dim,
                              self.n_layers,
                              bidirectional=self.bidirectional,
                              dropout=self.dropout,
                              batch_first=True)
        else:
            self.layer = nn.RNN(self.input_size,
                                self.encoder_dim,
                                self.n_layers,
                                bidirectional=self.bidirectional,
                                dropout=self.dropout,
                                batch_first=True)

    @staticmethod
    def get_required_params():
        return {
            'input_size': int,
            'encoder_dim': int,
        }

    @staticmethod
    def get_optional_params():
        return {'layer': str, 'n_layers': int, 'dropout': float, 'is_bidirectional': bool}

    def forward(self, inp, previous_hidden=None, pack=True):
        """
        inp: shape batch_size, seq_len, input_size
        previous_hidden: if given shape n_layers * num_directions,
        batch_size, embedding_dim.
               Initialized automatically if None
        return: embedded
        """
        input_tensor = inp[0]
        input_length = inp[1]
        batch_size = input_tensor.size(0)
        # TODO: warning: output shape is changed! (batch_first=True) Check hidden
        if pack:
            input_lengths_sorted, perm_idx = torch.sort(input_length, dim=0, descending=True)
            input_lengths_sorted = input_lengths_sorted.detach().to(device="cpu").tolist()
            input_tensor = torch.index_select(input_tensor, 0, perm_idx)
            rnn_input = pack_padded_sequence(input=input_tensor,
                                             lengths=input_lengths_sorted,
                                             batch_first=True)
        else:
            rnn_input = input_tensor
        if previous_hidden is None:
            previous_hidden = self.init_hidden(batch_size)
            if self.layer == 'LSTM':
                cell = self.init_cell(batch_size)
                previous_hidden = (previous_hidden, cell)
        else:
            if self.layer == 'LSTM':
                hidden = previous_hidden[0]
                cell = previous_hidden[1]
                hidden = torch.index_select(hidden, 1, perm_idx)
                cell = torch.index_select(cell, 1, perm_idx)
                previous_hidden = (hidden, cell)
            else:
                previous_hidden = torch.index_select(previous_hidden, 1, perm_idx)
        rnn_output, next_hidden = self.rnn(rnn_input)  # , previous_hidden)

        if pack:
            rnn_output, _ = pad_packed_sequence(rnn_output, batch_first=True)
            _, unperm_idx = perm_idx.sort(0)
            rnn_output = torch.index_select(rnn_output, 0, unperm_idx)
            if self.layer == 'LSTM':
                hidden = next_hidden[0]
                cell = next_hidden[1]
                hidden = torch.index_select(hidden, 1, unperm_idx)
                cell = torch.index_select(cell, 1, unperm_idx)
                next_hidden = (hidden, cell)
            else:
                next_hidden = torch.index_select(next_hidden, 1, unperm_idx)

        index_t = (input_length - 1).to(dtype=torch.long)
        index_t = index_t.view(-1, 1, 1).expand(-1, 1, rnn_output.size(2))

        embedded = torch.gather(rnn_output, dim=1, index=index_t).squeeze(1)

        return embedded, next_hidden

    def init_hidden(self, batch_size):
        if self.use_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        return torch.zeros(self.n_layers * self.n_directions, batch_size, self.encoder_dim, device=device)

    def init_cell(self, batch_size):
        if self.use_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        return torch.zeros(self.n_layers * self.n_directions, batch_size, self.encoder_dim, device=device)

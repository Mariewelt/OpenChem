# GRU_plain module from original GraphRNN implementation
# https://github.com/JiaxuanYou/graph-generation

import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.init as init


class GRUPlain(nn.Module):
    def __init__(self,
                 input_size,
                 embedding_size,
                 hidden_size,
                 num_layers,
                 has_input=True,
                 has_output=False,
                 has_output_nonlin=False,
                 output_size=None):
        super(GRUPlain, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.has_input = has_input
        self.has_output = has_output

        if has_input:
            # TODO: use small embedding layer here for edge class
            self.input = nn.Linear(input_size, embedding_size)
            self.rnn = nn.GRU(input_size=embedding_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        if has_output and has_output_nonlin:
            self.output = nn.Sequential(nn.Linear(hidden_size, embedding_size), nn.ReLU(),
                                        nn.Linear(embedding_size, output_size), nn.ReLU())
        elif has_output:
            self.output = nn.Sequential(nn.Linear(hidden_size, embedding_size), nn.ReLU(),
                                        nn.Linear(embedding_size, output_size))

        self.relu = nn.ReLU()
        # initialize
        self.hidden = None  # need initialize before forward run

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.25)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('sigmoid'))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    def forward(self, input_raw, pack=False, input_len=None, return_output_raw=False, enforce_sorted=True):
        if self.has_input:
            input = self.input(input_raw)
            input = self.relu(input)
        else:
            input = input_raw
        if pack:
            input = pack_padded_sequence(input, input_len, batch_first=True, enforce_sorted=enforce_sorted)
        output_raw, self.hidden = self.rnn(input, self.hidden)
        if pack:
            output_raw = pad_packed_sequence(output_raw, batch_first=True)[0]
        if self.has_output:
            output = self.output(output_raw)
        else:
            output = output_raw
        # return hidden state at each time step
        if return_output_raw:
            return output, output_raw
        else:
            return output

import torch.nn as nn
import torch.nn.functional as F
from openchem.modules.encoders.openchem_encoder import OpenChemEncoder
from openchem.modules.encoders.rnn_encoder import RNNEncoder

from openchem.utils.utils import check_params

from openchem.layers.conv_bn_relu import ConvBNReLU


class CNNEncoder(OpenChemEncoder):
    """Convolutional encoder"""

    def __init__(self, params, use_cuda):
        super(CNNEncoder, self).__init__(params, use_cuda)
        check_params(params, self.get_required_params(),
                     self.get_optional_params())
        self.dropout = params['dropout']
        self.input_size = params['input_size']
        self.encoder_dim = params['encoder_dim']
        self.pooling = params['pooling']
        if self.pooling not in ['max', 'mean', 'sum']:
            raise ValuError("Pooling must be one of 'max', 'mean', 'sum'")
        kernel_sizes = params['kernel_sizes']
        assert len(kernel_sizes) > 0
        self.convolutions = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        out_channels = self.encoder_dim
        for i in range(len(kernel_sizes)):
            if i == 0:
                in_channels = self.input_size
            else:
                in_channels = self.encoder_dim
            self.dropouts.append(nn.Dropout(self.dropout))
            kernel_size = kernel_sizes[i]
            self.convolutions.append(ConvBNReLU(in_channels=in_channels,
                                                out_channels=out_channels,
                                                kernel_size=kernel_size,
                                                stride=1))
        #rnn_params = {'layer': 'LSTM', 'input_size': out_channels,
        #              'encoder_dim': self.encoder_dim, 'n_layers': 1,
        #              'is_bidirectional': False, 'dropout': 0.0}
        #self.rnn = RNNEncoder(rnn_params, use_cuda)

    @staticmethod
    def get_required_params():
        return {
            'kernel_sizes': list,
            'encoder_dim': int,
        }

    @staticmethod
    def get_optional_params():
        return{
            'dropout': float
        }

    def forward(self, inp):
        x = inp[0]
        x = x.transpose(1, 2)
        for i in range(len(self.convolutions)):
            conv = self.convolutions[i]
            x = self.dropouts[i](x)
            x = conv(x)
            #x = F.relu(x)
            #x = x.squeeze(3)
        #print(x.size())
        #x, tmp = self.rnn((x.transpose(1, 2), inp_len))
        if self.pooling == 'mean':
            x = x.mean(dim=2)
        elif self.pooling == 'max':
            x, _ = x.max(dim=2)
        elif self.pooling == 'sum':
            x = x.sum(dim=2)
        return x, 0

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
        self.encoder_dim = params['encoder_dim']
        convolutions = params['convolutions']
        assert len(convolutions) > 0
        self.convolutions = nn.ModuleList()
        for i in range(len(convolutions)):
            in_channels = convolutions[i][0]
            kernel_size = convolutions[i][1]
            self.convolutions.append(ConvBNReLU(in_channels=in_channels,
                                                out_channels=in_channels*2,
                                                kernel_size=kernel_size))
        rnn_params = {'layer': 'LSTM', 'input_size': in_channels,
                      'encoder_dim': self.encoder_dim, 'n_layers': 1,
                      'is_bidirectional': True, 'dropout': 0.0}
        self.rnn = RNNEncoder(rnn_params, use_cuda)

    @staticmethod
    def get_required_params():
        return {
            'convolutions': list,
            'encoder_dim': int,
            'dropout': float,
        }

    @staticmethod
    def get_optional_params():
        return{
            'dropout': float
        }

    def forward(self, x):
        x = x.transpose(1, 2)
        for i in range(len(self.convolutions)):
            conv = self.convolutions[i]
            if conv.kernel_size % 2 == 1:
                # padding is implicit in the conv
                x = conv(x)
            else:
                padding_l = (conv.kernel_size - 1) // 2
                padding_r = conv.kernel_size[0] // 2
                x = F.pad(x, (0, 0, 0, 0, padding_l, padding_r))
                x = conv(x)
            x = F.glu(x, dim=1)
        x = self.rnn(x.transpose(1, 2))
        return x

import torch
from torch import nn

from openchem.utils.utils import check_params


class OpenChemEncoder(nn.Module):
    """Base class for embedding module"""
    def __init__(self, params, use_cuda=None):
        super(OpenChemEncoder, self).__init__()
        check_params(params, self.get_required_params(), self.get_required_params())
        self.params = params
        if use_cuda is None:
            use_cuda = torch.cuda.is_available()
        self.use_cuda = use_cuda
        self.input_size = self.params['input_size']
        self.encoder_dim = self.params['encoder_dim']

    @staticmethod
    def get_required_params():
        return {'input_size': int, 'encoder_dim': int}

    @staticmethod
    def get_optional_params():
        return {}

    def forward(self, inp):
        raise NotImplementedError

import torch.nn as nn

from openchem.utils.utils import check_params


class OpenChemEmbedding(nn.Module):
    def __init__(self, params):
        super(OpenChemEmbedding, self).__init__()
        check_params(params, self.get_required_params(), self.get_optional_params())
        self.params = params
        self.num_embeddings = self.params['num_embeddings']
        if 'padding_idx' in params.keys():
            self.padding_idx = self.params['padding_idx']
        else:
            self.padding_idx = None

    def forward(self, inp):
        raise NotImplementedError

    @staticmethod
    def get_required_params():
        return {
            'num_embeddings': int,
        }

    @staticmethod
    def get_optional_params():
        return {'padding_idx': int}

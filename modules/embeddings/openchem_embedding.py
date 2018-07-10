import torch.nn as nn

from utils.utils import check_params


class OpenChemEmbedding(nn.Module):
    def __init__(self, params):
        super(OpenChemEmbedding, self).__init__()
        check_params(params, self.get_params())
        self.params = params
        self.num_embeddings = self.params['num_embeddings']
        self.embedding_dim = self.params['embedding_dim']
        self.padding_idx = self.params['padding_idx']
        self.embedding = nn.Embedding(num_embeddings=self.num_embeddings,
                                      embedding_dim=self.embedding_dim,
                                      padding_idx=self.padding_idx)

    def forward(self, inp):
        raise NotImplementedError

    @staticmethod
    def get_params():
        return {
            'num_embeddings': int,
            'embedding_dim': int,
            'padding_idx': int
        }

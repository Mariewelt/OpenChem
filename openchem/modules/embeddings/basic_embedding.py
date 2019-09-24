from openchem.modules.embeddings.openchem_embedding import OpenChemEmbedding

from torch import nn


class Embedding(OpenChemEmbedding):
    def __init__(self, params):
        super(Embedding, self).__init__(params)
        self.embedding_dim = self.params['embedding_dim']
        self.embedding = nn.Embedding(num_embeddings=self.num_embeddings,
                                      embedding_dim=self.embedding_dim,
                                      padding_idx=self.padding_idx)

    def forward(self, inp):
        embedded = self.embedding(inp)
        return embedded

    @staticmethod
    def get_required_params():
        return {
            'embedding_dim': int,
        }

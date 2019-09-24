import torch
from openchem.modules.embeddings.openchem_embedding import OpenChemEmbedding


class OneHotEmbedding(OpenChemEmbedding):
    def __init__(self, params):
        super(OneHotEmbedding, self).__init__(params)
        if self.padding_idx is not None:
            weight = torch.eye(self.num_embeddings - 1)
            zero_row = torch.zeros(self.num_embeddings - 1).unsqueeze(0)
            weight = torch.cat([weight[:self.padding_idx], zero_row, weight[self.padding_idx:]], dim=0)
        else:
            weight = torch.eye(self.num_embeddings)
        self.weight = torch.nn.Parameter(weight, requires_grad=False)

    def forward(self, inp):
        embedded = self.weight[inp]
        return embedded

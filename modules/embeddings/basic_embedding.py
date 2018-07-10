from modules.embeddings.openchem_embedding import OpenChemEmbedding


class Embedding(OpenChemEmbedding):
    def __init__(self, params):
        super(Embedding, self).__init__(params)

    def forward(self, inp):
        embedded = self.embedding(inp)
        return embedded

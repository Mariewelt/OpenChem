from modules.embeddings.openchem_embedding import OpenChemEmbedding

from utils import utils


class PositionalEmbedding(OpenChemEmbedding):
    def __init__(self, params):
        super(PositionalEmbedding, self).__init__(params)
        self.left_pad = params['left_pad']

    def forward(self, inp, incremental_state=None):
        """Input is expected to be of size [bsz x seq_len]."""
        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            positions = inp.data.new(1, 1).fill_(
                self.padding_idx + inp.size(1))
        else:
            positions = utils.make_positions(inp.data, self.padding_idx,
                                             self.left_pad)
        embedded = self.embedding(positions)

        return embedded

    def max_positions(self):
        """Maximum number of supported positions."""

        return self.num_embeddings - self.padding_idx - 1

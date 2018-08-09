from openchem.models.openchem_model import OpenChemModel
from openchem.optimizer.openchem_optimizer import OpenChemOptimizer
from openchem.optimizer.openchem_lr_scheduler import OpenChemLRScheduler
from openchem.data.utils import cut_padding


import torch


class MoleculeProtein2Label(OpenChemModel):
    r"""
    Creates a model that predicts one or multiple labels given two sequences as
    input. Embeddings for each input are extracted separately with Embedding
    layer, followed by encoder (could be RNN or CNN encoder) and then merged
    together. Last layer of the model is multi-layer perceptron.

    Args:
        params (dict): dictionary describing model architecture.

    """
    def __init__(self, params):
        super(MoleculeProtein2Label, self).__init__(params)
        self.mol_embedding = self.params['mol_embedding']
        self.mol_embed_params = self.params['mol_embedding_params']
        self.prot_embedding = self.params['prot_embedding']
        self.prot_embed_params = self.params['prot_embedding_params']
        self.MolEmbedding = self.mol_embedding(self.mol_embed_params)
        self.ProtEmbedding = self.prot_embedding(self.prot_embed_params)
        self.mol_encoder = self.params['mol_encoder']
        self.mol_encoder_params = self.params['mol_encoder_params']
        self.prot_encoder = self.params['prot_encoder']
        self.prot_encoder_params = self.params['prot_encoder_params']
        self.MolEncoder = self.mol_encoder(self.mol_encoder_params,
                                           self.use_cuda)
        self.ProtEncoder = self.prot_encoder(self.prot_encoder_params,
                                             self.use_cuda)
        self.merge = self.params['merge']
        self.mlp = self.params['mlp']
        self.mlp_params = self.params['mlp_params']
        self.MLP = self.mlp(self.mlp_params)

    def forward(self, inp, eval=False):
        if eval:
            self.eval()
        else:
            self.train()
        mol = inp[0]
        prot = inp[1]
        mol_embedded = self.MolEmbedding(mol)
        mol_output, _ = self.MolEncoder(mol_embedded)
        prot_embedded = self.ProtEmbedding(prot)
        prot_output, _ = self.ProtEncoder(prot_embedded)
        if self.merge == 'mul':
            output = mol_output*prot_output
        elif self.merge == 'concat':
            output = torch.cat((mol_output, prot_output), 1)
        else:
            raise ValueError('Invalid value for merge')
        output = self.MLP(output)
        return output

    def cast_inputs(self, sample):
        batch_mols = cut_padding(sample['tokenized_smiles'], sample['mol_length'],
                                  padding='left')
        batch_prots = cut_padding(sample['tokenized_protein'], sample['prot_length'],
                                  padding='left')
        batch_mols = torch.tensor(batch_mols, requires_grad=True).long()
        batch_prots = torch.tensor(batch_prots, requires_grad=True).long()
        batch_labels = torch.tensor(sample['labels'])
        if self.task == 'classification':
            batch_labels = batch_labels.long()
        elif self.task == 'regression':
            batch_labels = batch_labels.float()
        if self.use_cuda:
            batch_mols = batch_mols.cuda()
            batch_prots = batch_prots.cuda()
            batch_labels = batch_labels.cuda()
        return (batch_mols, batch_prots), batch_labels

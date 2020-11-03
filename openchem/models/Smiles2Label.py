from openchem.models.openchem_model import OpenChemModel
from openchem.optimizer.openchem_optimizer import OpenChemOptimizer
from openchem.optimizer.openchem_lr_scheduler import OpenChemLRScheduler

import torch


class Smiles2Label(OpenChemModel):
    r"""
    Creates a model that predicts one or multiple labels given string of
    characters as input. Embeddings for input sequences are extracted with
    Embedding layer, followed by encoder (could be RNN or CNN encoder).
    Last layer of the model is multi-layer perceptron.

    Args:
        params (dict): dictionary describing model architecture.

    """
    def __init__(self, params):
        super(Smiles2Label, self).__init__(params)
        self.embedding = self.params['embedding']
        self.embed_params = self.params['embedding_params']
        self.Embedding = self.embedding(self.embed_params)
        self.encoder = self.params['encoder']
        self.encoder_params = self.params['encoder_params']
        self.Encoder = self.encoder(self.encoder_params, self.use_cuda)
        self.mlp = self.params['mlp']
        self.mlp_params = self.params['mlp_params']
        self.MLP = self.mlp(self.mlp_params)
        self.optimizer = OpenChemOptimizer([self.params['optimizer'], self.params['optimizer_params']],
                                           self.parameters())
        self.scheduler = OpenChemLRScheduler([self.params['lr_scheduler'], self.params['lr_scheduler_params']],
                                             self.optimizer.optimizer)

    def forward(self, inp, eval=False):
        if eval:
            self.eval()
        else:
            self.train()
        input_tensor = inp[0]
        input_length = inp[1]
        embedded = self.Embedding(input_tensor)
        output, _ = self.Encoder([embedded, input_length])
        output = self.MLP(output)
        return output

    @staticmethod
    def cast_inputs(sample, task, use_cuda, for_prediction=False):
        batch_mols = sample['tokenized_smiles'].to(dtype=torch.long)
        if for_prediction and "object" in sample.keys():
            batch_object = sample['object']
        else:
            batch_object = None
        batch_length = sample['length'].to(dtype=torch.long)
        if not for_prediction and "labels" in sample.keys():
            batch_labels = sample['labels'].to(dtype=torch.float)
            if task == 'classification':
                batch_labels = batch_labels.to(dtype=torch.long)
        else:
            batch_labels = None
        if use_cuda:
            batch_mols = batch_mols.to(device="cuda")
            batch_length = batch_length.to(device="cuda")
            if batch_labels is not None:
                batch_labels = batch_labels.to(device="cuda")
        if batch_object is not None:
            return (batch_mols, batch_length), batch_object
        else:
            return (batch_mols, batch_length), batch_labels

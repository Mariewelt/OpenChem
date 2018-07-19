from models.openchem_model import OpenChemModel
from optimizer.openchem_optimizer import OpenChemOptimizer
from optimizer.openchem_lr_scheduler import OpenChemLRScheduler

import torch


class Smiles2Label(OpenChemModel):
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
        self.optimizer = OpenChemOptimizer([self.params['optimizer'],
                                            self.params['optimizer_params']],
                                           self.parameters())
        self.scheduler = OpenChemLRScheduler([self.params['lr_scheduler'],
                                              self.params['lr_scheduler_params']
                                              ],
                                             self.optimizer.optimizer)

    def forward(self, inp, eval=False):
        if eval:
            self.eval()
        else:
            self.train()
        embedded = self.Embedding(inp)
        output = self.Encoder(embedded)
        output = self.MLP(output)
        return output

    def cast_inputs(self, sample):
        batch_mols = torch.tensor(sample['tokenized_smiles'],
                                  requires_grad=True).long()
        batch_labels = torch.tensor(sample['labels']).float()
        if self.task == 'classification':
            batch_labels = batch_labels.long()
        if self.use_cuda:
            batch_mols = batch_mols.cuda()
            batch_labels = batch_labels.cuda()
        return batch_mols, batch_labels

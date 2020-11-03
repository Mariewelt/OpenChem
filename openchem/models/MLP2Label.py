from openchem.models.openchem_model import OpenChemModel
from openchem.optimizer.openchem_optimizer import OpenChemOptimizer
from openchem.optimizer.openchem_lr_scheduler import OpenChemLRScheduler

import torch


class MLP2Label(OpenChemModel):
    r"""
    Creates a model that predicts one or multiple labels given a numerical feature vector.

    Args:
        params (dict): dictionary describing model architecture.

    """
    def __init__(self, params):
        super(MLP2Label, self).__init__(params)
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
        output = self.MLP(inp)
        return output

    @staticmethod
    def cast_inputs(sample, task, use_cuda, for_prediction=False):
        batch_mols = sample['features']
        if for_prediction and "object" in sample.keys():
            batch_object = sample['object']
        else:
            batch_object = None
        if not for_prediction and "labels" in sample.keys():
            batch_labels = sample['labels'].to(dtype=torch.float)
            if task == 'classification':
                batch_labels = batch_labels.to(dtype=torch.long)
        else:
            batch_labels = None
        if use_cuda:
            batch_mols = batch_mols.to(device="cuda")
            if batch_labels is not None:
                batch_labels = batch_labels.to(device="cuda")
        if batch_object is not None:
            return batch_mols, batch_object
        else:
            return batch_mols, batch_labels

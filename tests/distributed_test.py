from models.openchem_model import OpenChemModel
import torch
from torch.nn.utils import clip_grad_norm_


class DistributedTest(OpenChemModel):
    def __init__(self, params):
        super(DistributedTest, self).__init__(params)
        self.input_size = params['input_size']
        self.w = torch.nn.Parameter(torch.rand_like(torch.zeros(self.input_size, 1)))

    def forward(self, inp, eval=False):
        output = torch.mm(inp, self.w)
        return output

    def cast_inputs(self, sample):
        batch_mols = torch.tensor(sample['tokenized_smiles'],
                                  requires_grad=True).float()
        batch_labels = torch.tensor(sample['labels']).float()
        if self.task == 'classification':
            batch_labels = batch_labels.long()
        if self.use_cuda:
            batch_mols = batch_mols.cuda()
            batch_labels = batch_labels.cuda()
        return batch_mols, batch_labels


def train_step(model, optimizer, criterion, inp, target):
    optimizer.zero_grad()
    output = model.forward(inp, eval=False)
    print(loss.)
    optimizer.step()
    if model.module.use_clip_grad:
        clip_grad_norm_(model.parameters(), model.module.max_grad_norm)

    return None



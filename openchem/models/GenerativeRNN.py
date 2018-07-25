from openchem.models.openchem_model import OpenChemModel
from openchem.layers.stack_augmentation import StackAugmentation
from openchem.optimizer.openchem_optimizer import OpenChemOptimizer
from openchem.optimizer.openchem_lr_scheduler import OpenChemLRScheduler

import torch
import torch.nn.functional as F


class GenerativeRNN(OpenChemModel):
    def __init__(self, params):
        super(GenerativeRNN, self).__init__(params)
        self.has_cell = True
        self.has_stack = True
        self.input_size = params['input_size']
        self.hidden_size = params['hidden_size']
        self.output_size = params['output_size']
        self.has_stack = params['has_stack']
        if self.has_stack:
            self.Stack = StackAugmentation(**self.params['stack_params'])
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

    def forward(self, inp_seq):
        """Generator forward function."""
        output = torch.zeros(inp_seq.size()[0])
        hidden = None
        for c in range(len(inp_seq)):
            inp_token = self.Embedding(inp_seq[c].view(1, -1))
            if self.module.has_stack:
                stack_controls = self.stack_controls_layer(hidden.squeeze(0))
                stack_controls = F.softmax(stack_controls, dim=1)
                stack_input = self.stack_input_layer(hidden)
                stack_input = F.tanh(stack_input)
                stack = self.stack_augmentation(stack_input.permute(1, 0, 2),
                                                stack, stack_controls)
                stack_top = stack[:, 0, :].unsqueeze(0)
                inp_token = torch.cat((inp_token, stack_top), dim=2)
            output, hidden = self.Encoder(inp_token.view(1, 1, -1),
                                          hidden)
            output[c] = self.decoder(output.view(1, -1))

        return output

    def cast_inputs(self, sample):
        seq = torch.tensor(sample['seq'],
                               requires_grad=True).long()
        target = torch.tensor(sample['target']).long()
        return seq, target

from openchem.models.openchem_model import OpenChemModel
from openchem.layers.stack_augmentation import StackAugmentation

import torch
import torch.nn.functional as F


class GenerativeRNN(OpenChemModel):
    def __init__(self, params):
        super(GenerativeRNN, self).__init__(params)
        self.has_stack = params['has_stack']
        if self.has_stack:
            self.Stack = StackAugmentation(use_cuda=self.use_cuda,
                                           **self.params['stack_params'])
        self.embedding = self.params['embedding']
        self.embed_params = self.params['embedding_params']
        self.Embedding = self.embedding(self.embed_params)
        self.encoder = self.params['encoder']
        self.encoder_params = self.params['encoder_params']
        self.Encoder = self.encoder(self.encoder_params, self.use_cuda)
        self.mlp = self.params['mlp']
        self.mlp_params = self.params['mlp_params']
        self.MLP = self.mlp(self.mlp_params)

    def forward(self, inp_seq, eval=False):
        """Generator forward function."""
        if eval:
            self.eval()
        else:
            self.train()
        batch_size = inp_seq.size()[0]
        seq_len = inp_seq.size()[1]
        result = torch.zeros(batch_size, seq_len, self.MLP.hidden_size[-1], requires_grad=True)
        if self.use_cuda:
            result = result.cuda()
        hidden = self.Encoder.init_hidden(batch_size)
        if self.has_stack:
            stack = self.Stack.init_stack(batch_size)
        for c in range(seq_len):
            inp_token = self.Embedding(inp_seq[:, c].view(batch_size, -1))
            if self.has_stack:
                stack = self.Stack(hidden, stack)
                stack_top = stack[:, 0, :].unsqueeze(1)
                inp_token = torch.cat((inp_token, stack_top), dim=2)
            output, hidden = self.Encoder(inp_token, hidden)
            result[:, c, :] = self.MLP(output)
        
        n_classes = result.size()[2] 
        return result.view(-1, n_classes)

    def cast_inputs(self, sample):
        sample_seq = sample['tokenized_smiles']
        lengths = sample['length']
        max_len = lengths.max(dim=0)[0].cpu().numpy()
        batch_size = len(lengths)
        sample_seq = sample_seq[:, :max_len]
        target = sample_seq[:, 1:, 0].contiguous().view((batch_size*(max_len-1), 1))
        seq = sample_seq[:, :-1, 0]
        seq = torch.tensor(seq, requires_grad=True).long()
        target = torch.tensor(target).long()
        seq = seq.cuda()
        target = target.cuda()
        return seq, target.squeeze(1)

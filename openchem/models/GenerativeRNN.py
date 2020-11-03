from openchem.models.openchem_model import OpenChemModel
from openchem.layers.stack_augmentation import StackAugmentation
from openchem.data.utils import seq2tensor, cut_padding

import torch
import numpy as np


class GenerativeRNN(OpenChemModel):
    def __init__(self, params):
        super(GenerativeRNN, self).__init__(params)
        self.has_stack = params['has_stack']
        if self.has_stack:
            self.Stack = StackAugmentation(use_cuda=self.use_cuda, **self.params['stack_params'])
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
        n_classes = self.MLP.hidden_size[-1]
        result = torch.zeros(batch_size, seq_len, n_classes, requires_grad=True)
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

        return result.view(-1, n_classes)

    def infer(self, prime_str, n_to_generate, max_len, tokens, temperature=0.8):
        self.eval()
        tokens = np.array(tokens).reshape(-1)
        prime_str = [prime_str] * n_to_generate
        tokens = list(tokens[0])
        num_tokens = len(tokens)
        prime_input = seq2tensor(prime_str, tokens)
        tokens = np.array(tokens)
        batch_size = prime_input.shape[0]
        seq_len = prime_input.shape[1] - 1
        hidden = self.Encoder.init_hidden(batch_size)
        prime_input = torch.tensor(prime_input).long()
        if self.use_cuda:
            prime_input = prime_input.cuda()
        if self.has_stack:
            stack = self.Stack.init_stack(batch_size)
        for c in range(seq_len):
            inp_token = self.Embedding(prime_input[:, c].view(batch_size, -1))
            if self.has_stack:
                stack = self.Stack(hidden, stack)
                stack_top = stack[:, 0, :].unsqueeze(1)
                inp_token = torch.cat((inp_token, stack_top), dim=2)
            output, hidden = self.Encoder(inp_token, hidden)
        inp = prime_input[:, -1]
        predicted = [' '] * (batch_size * (max_len - seq_len))
        predicted = np.reshape(predicted, (batch_size, max_len - seq_len))
        for c in range(max_len - seq_len):
            inp_token = self.Embedding(inp.view(batch_size, -1))
            if self.has_stack:
                stack = self.Stack(hidden, stack)
                stack_top = stack[:, 0, :].unsqueeze(1)
                inp_token = torch.cat((inp_token, stack_top), dim=2)
            output, hidden = self.Encoder(inp_token, hidden)
            output = self.MLP(output)
            output_dist = output.data.view(-1).div(temperature).exp()
            output_dist = output_dist.view(batch_size, num_tokens)
            top_i = torch.multinomial(output_dist, 1)
            # Add predicted character to string and use as next input
            predicted_char = tokens[top_i]
            predicted[:, c] = predicted_char[:, 0]
            inp = torch.tensor(top_i)

        return predicted

    @staticmethod
    def cast_inputs(sample, task, use_cuda):
        sample_seq = sample['tokenized_smiles']
        lengths = sample['length']
        max_len = lengths.max(dim=0)[0].cpu().numpy()
        batch_size = len(lengths)
        sample_seq = cut_padding(sample_seq, lengths, padding='right')
        target = sample_seq[:, 1:].contiguous().view((batch_size * (max_len - 1), 1))
        seq = sample_seq[:, :-1]
        seq = torch.tensor(seq, requires_grad=True).long()
        target = torch.tensor(target).long()
        seq = seq.cuda()
        target = target.cuda()
        return seq, target.squeeze(1)

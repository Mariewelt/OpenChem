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
        if self.encoder_params['is_bidirectional']:
            raise ValueError('RNN cannot be bidirectional in GenerativeRNN '
                             'model')
        self.Encoder = self.encoder(self.encoder_params, self.use_cuda)
        self.mlp = self.params['mlp']
        self.mlp_params = self.params['mlp_params']
        self.MLP = self.mlp(self.mlp_params)

    def forward_step(self, inp, hidden, stack):
        batch_size = len(inp)
        inp = self.Embedding(inp)

        if self.has_stack:
            if self.encoder_params['layer'] == 'LSTM':
                hidden_ = hidden[0]
            else:
                hidden_ = hidden
            hidden_2_stack = hidden_.squeeze(0)
            stack = self.Stack(hidden_2_stack, stack)
            stack_top = stack[:, 0, :].unsqueeze(1)
            inp = torch.cat((inp.unsqueeze(1), stack_top), dim=2)
        else:
            inp = inp.unsqueeze(1)
        inp_length = torch.tensor(np.array([1] * batch_size)).long()
        if self.use_cuda:
            inp_length = inp_length.cuda()
        inp = [inp, inp_length]
        output, hidden = self.Encoder(inp, hidden)
        output = self.MLP(output)
        return output, hidden, stack

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

            output, hidden, stack = self.forward_step(inp_seq[:, c],
                                                      hidden,
                                                      stack)
            result[:, c, :] = output

        return result.view(-1, n_classes)

    def infer(self, batch_size, prime_str, tokens, max_len=120, end_token='>'):
        self.eval()
        hidden = self.Encoder.init_hidden(batch_size)
        if self.has_stack:
            stack = self.Stack.init_stack(batch_size)
        prime_input, _ = seq2tensor([prime_str] * batch_size,
                                         tokens=tokens,
                                         flip=False)
        prime_input = torch.tensor(prime_input).long()
        if self.use_cuda:
            prime_input = prime_input.cuda()
        new_samples = [[prime_str] * batch_size]

        # Use priming string to "build up" hidden state
        for p in range(len(prime_str[0]) - 1):
            _, hidden, stack = self.forward_step(prime_input[:, p],
                                                 hidden,
                                                 stack)
        inp = prime_input[:, -1]

        for p in range(max_len):
            output, hidden, stack = self.forward_step(inp, hidden, stack)
            # Sample from the network as a multinomial distribution
            probs = torch.softmax(output, dim=1).detach()
            top_i = torch.multinomial(probs, 1).cpu().numpy()

            # Add predicted character to string and use as next input
            predicted_char = (np.array(tokens)[top_i].reshape(-1))
            predicted_char = predicted_char.tolist()
            new_samples.append(predicted_char)

            # Prepare next input token for the generator
            inp, _ = seq2tensor(predicted_char, tokens=tokens)
            inp = torch.tensor(inp.squeeze(1)).long()
            if self.use_cuda:
                inp = inp.cuda()

        # Remove characters after end tokens
        string_samples = []
        new_samples = np.array(new_samples)
        #print(new_samples)
        for i in range(batch_size):
            sample = list(new_samples[:, i])
            if end_token in sample:
                end_token_idx = sample.index(end_token)
                string_samples.append(''.join(sample[1:end_token_idx]))
        return string_samples

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

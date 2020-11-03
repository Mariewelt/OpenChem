from openchem.models.openchem_model import OpenChemModel
from openchem.optimizer.openchem_optimizer import OpenChemOptimizer
from openchem.optimizer.openchem_lr_scheduler import OpenChemLRScheduler
from openchem.data.utils import cut_padding
from openchem.models.Graph2Label import Graph2Label
from openchem.models.Smiles2Label import Smiles2Label


import torch


class SiameseModel(OpenChemModel):
    r"""
    Creates a model that predicts one or multiple labels given two sequences as
    input. Embeddings for each input are extracted separately with Embedding
    layer, followed by encoder (could be RNN or CNN encoder) and then merged
    together. Last layer of the model is multi-layer perceptron.

    Args:
        params (dict): dictionary describing model architecture.

    """
    def __init__(self, params):
        super(SiameseModel, self).__init__(params)
        self.head1_embedding = self.params['head1_embedding']
        self.head1_embed_params = self.params['head1_embedding_params']
        self.head2_embedding = self.params['head2_embedding']
        self.head2_embed_params = self.params['head2_embedding_params']
        if self.head1_embedding is not None:
            self.Head1Embedding = self.head1_embedding(self.head1_embed_params)
        if self.head2_embedding is not None:
            self.Head2Embedding = self.head2_embedding(self.head2_embed_params)
        self.head1_encoder = self.params['head1_encoder']
        self.head1_encoder_params = self.params['head1_encoder_params']
        self.head2_encoder = self.params['head2_encoder']
        self.head2_encoder_params = self.params['head2_encoder_params']
        self.Head1Encoder = self.head1_encoder(self.head1_encoder_params,
                                           self.use_cuda)
        self.Head2Encoder = self.head2_encoder(self.head2_encoder_params,
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
        head1 = inp[0]
        if self.Head1Embedding is not None:
            head1_input = head1[0]
            head1_length = head1[1]
        else:
            head1_input = head1
        head2 = inp[1]
        if self.Head1Embedding is not None:
            head2_input = head2[0]
            head2_length = head2[1]
        else:
            head2_input = head2
        if self.Head1Embedding is not None:
            head1_embedded = self.Head1Embedding(head1_input)
            head1_embedded = [head1_embedded, head1_length]
        else:
            head1_embedded = head1_input
        if self.Head2Embedding is not None:
            head2_embedded = self.Head2Embedding(head2_input)
            head2_embedded = [head2_embedded, head2_length]
        else:
            head2_embedded = head2_input
        head1_output, _ = self.Head1Encoder(head1_embedded)
        head2_output, _ = self.Head2Encoder(head2_embedded)
        if self.merge == 'mul':
            output = head1_output*head2_output
        elif self.merge == 'concat':
            output = torch.cat((head1_output, head2_output), 1)
        else:
            raise ValueError('Invalid value for merge')
        output = self.MLP(output)
        return output

    @staticmethod
    def cast_inputs(sample, task, use_cuda, for_prediction=False):
        if 'tokenized_smiles' in sample['head1'].keys():
            batch_head1, batch_labels1 = Smiles2Label.cast_inputs(sample['head1'],
                                                                 task,
                                                                 use_cuda,
                                                                 for_prediction)
        else:
            batch_head1, batch_labels1 = Graph2Label.cast_inputs(sample['head1'],
                                                                task,
                                                                use_cuda,
                                                                for_prediction)
        if 'tokenized_smiles' in sample['head2'].keys():
            batch_head2, batch_labels2 = Smiles2Label.cast_inputs(sample['head2'],
                                                                 task,
                                                                 use_cuda,
                                                                 for_prediction)
        else:
            batch_head2, batch_labels2 = Graph2Label.cast_inputs(sample['head2'],
                                                                task,
                                                                use_cuda,
                                                                for_prediction)
        if for_prediction:
            batch_labels = batch_labels1 + [ord(",")] + batch_labels2
        else:
            batch_labels = sample["labels"]
            if task == "classification":
                batch_labels = batch_labels.to(dtype=torch.long)
        if use_cuda:
            batch_labels = batch_labels.to(device="cuda")
        return (batch_head1, batch_head2), batch_labels

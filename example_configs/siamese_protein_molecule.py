from openchem.models.SiameseModel import SiameseModel
from openchem.modules.embeddings.basic_embedding import Embedding
from openchem.modules.encoders.rnn_encoder import RNNEncoder
from openchem.modules.mlp.openchem_mlp import  OpenChemMLPSimple
from openchem.data.siamese_data_layer import SiameseDataset
from openchem.utils.utils import identity

import torch.nn as nn

from torch.optim import  Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from sklearn.metrics import  r2_score


head1_arguments = {
    "delimiter": ",",
    "sanitize": False,
}
head2_arguments = {
    "delimiter": ",",
    "sanitize": False
}
train_dataset = SiameseDataset('./benchmark_datasets/3familiy_with_mutations_with_embeddings.csv',
                               head1_type='protein_seq', head2_type='mol_smiles', cols_to_read=[34, 12, 1],
                               head1_arguments=head1_arguments, head2_arguments=head2_arguments,
                               )

#test_dataset = SiameseDataset('./benchmark_datasets/reactions/test.smi',
#                               head1_type='smiles', head2_type='smiles', cols_to_read=[0, 1, 2],
#                               head1_arguments=head1_arguments, head2_arguments=head2_arguments
#                               )

tokens_protein = train_dataset.head1_dataset.tokens
tokens_smiles = train_dataset.head2_dataset.tokens

print(len(tokens_protein))
print(len(tokens_smiles))

model = SiameseModel

model_params = {
    'use_cuda': True,
    'task': 'regression',
    'random_seed': 5,
    'use_clip_grad': True,
    'max_grad_norm': 10.0,
    'batch_size': 256,
    'num_epochs': 51,
    'logdir': './logs/test',
    'print_every': 1,
    'save_every': 1,
    'train_data_layer': train_dataset,
    'val_data_layer': None,
    'eval_metrics': r2_score,
    'criterion': nn.MSELoss(),
    'optimizer': Adam,
    'optimizer_params': {
        'lr': 0.001,
        },
    'lr_scheduler': StepLR,
    'lr_scheduler_params': {
        'step_size': 10,
        'gamma': 0.8
    },
    'head1_embedding': Embedding,
    'head1_embedding_params': {
        'num_embeddings': len(tokens_protein),
        'embedding_dim': 128,
        'padding_idx': tokens_protein.index(' ')
    },
    'head2_embedding': Embedding,
    'head2_embedding_params': {
        'num_embeddings': len(tokens_smiles),
        'embedding_dim': 128,
        'padding_idx': tokens_smiles.index(' ')
    },
    'head1_encoder': RNNEncoder,
    'head1_encoder_params': {
        'input_size': 128,
        'layer': "LSTM",
        'encoder_dim': 128,
        'n_layers': 4,
        'dropout': 0.8,
        'is_bidirectional': False
    },
    'head2_encoder': RNNEncoder,
    'head2_encoder_params': {
        'input_size': 128,
        'layer': "LSTM",
        'encoder_dim': 128,
        'n_layers': 4,
        'dropout': 0.8,
        'is_bidirectional': False
    },
    'merge': "mul",
    'mlp': OpenChemMLPSimple,
    'mlp_params': {
        'input_size': 128,
        'n_layers': 2,
        'hidden_size': [128, 1],
        'activation': [F.relu, identity],
    }
}

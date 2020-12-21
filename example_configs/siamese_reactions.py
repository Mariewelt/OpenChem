from openchem.models.SiameseModel import SiameseModel
from openchem.modules.embeddings.basic_embedding import Embedding
from openchem.modules.encoders.rnn_encoder import RNNEncoder
from openchem.modules.encoders.gcn_encoder import GraphCNNEncoder
from openchem.modules.mlp.openchem_mlp import OpenChemMLP, OpenChemMLPSimple
from openchem.data.siamese_data_layer import SiameseDataset
from openchem.utils.utils import identity

import torch
import torch.nn as nn

import numpy as np

from torch.optim import RMSprop, Adam
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score, f1_score

from openchem.data.utils import read_smiles_property_file
data = read_smiles_property_file('./benchmark_datasets/reactions/4_11_with_y2.csv',
                                 cols_to_read=[11, 12, 14],
                                 keep_header=False)
reactant1 = data[0]
reactant2 = data[1]
labels = np.array(data[2], dtype="float").reshape(-1, 1)

reactants = [reactant1[i] + " " + reactant2[i] for i in range(len(reactant2))]

from openchem.data.utils import get_tokens
tokens, _, _ = get_tokens(reactants)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(reactants, labels, test_size=0.2,
                                                    random_state=42)
y_mean = np.mean(y_train)
y_std = np.std(y_train)
y_train = (y_train - y_mean) / y_std
y_test = (y_test - y_mean) / y_std

from openchem.data.utils import save_smiles_property_file
save_smiles_property_file('./benchmark_datasets/reactions/train.smi', X_train, y_train, delimiter=" ")
save_smiles_property_file('./benchmark_datasets/reactions/test.smi', X_test, y_test, delimiter=" ")

from openchem.data.smiles_data_layer import SmilesDataset
head1_arguments = {
    "tokens": tokens,
    "delimiter": " ",
    "sanitize": False
}
head2_arguments = {
    "tokens": tokens,
    "delimiter": " ",
    "sanitize": False
}
train_dataset = SiameseDataset('./benchmark_datasets/reactions/train.smi',
                               head1_type='smiles', head2_type='smiles', cols_to_read=[0, 1, 2],
                               head1_arguments=head1_arguments, head2_arguments=head2_arguments
                               )

test_dataset = SiameseDataset('./benchmark_datasets/reactions/test.smi',
                               head1_type='smiles', head2_type='smiles', cols_to_read=[0, 1, 2],
                               head1_arguments=head1_arguments, head2_arguments=head2_arguments
                               )

model = SiameseModel

model_params = {
    'use_cuda': True,
    'task': 'classification',
    'random_seed': 5,
    'use_clip_grad': True,
    'max_grad_norm': 10.0,
    'batch_size': 256,
    'num_epochs': 51,
    'logdir': './logs/reactions',
    'print_every': 1,
    'save_every': 1,
    'train_data_layer': train_dataset,
    'val_data_layer': test_dataset,
    'eval_metrics': f1_score,
    'criterion': nn.CrossEntropyLoss(),#nn.MSELoss(),
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
        'num_embeddings': len(tokens),
        'embedding_dim': 128,
        'padding_idx': tokens.index(' ')
    },
    'head2_embedding': Embedding,
    'head2_embedding_params': {
        'num_embeddings': len(tokens),
        'embedding_dim': 128,
        'padding_idx': tokens.index(' ')
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
        'hidden_size': [128, 2],
        'activation': [F.relu, nn.Softmax(dim=1)],
    }
}

from openchem.models.GenerativeRNN import GenerativeRNN
from openchem.modules.embeddings.basic_embedding import Embedding
from openchem.modules.encoders.rnn_encoder import RNNEncoder
from openchem.modules.mlp.openchem_mlp import OpenChemMLP
from openchem.data.smiles_data_layer import SmilesDataset

import torch.nn as nn

from torch.optim import RMSprop
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F

train_dataset = SmilesDataset('./benchmark_datasets/Lipophilicity_dataset/Lipophilicity_train.csv',
                              cols_to_read=[0], pad=False, augment=False)
val_dataset = SmilesDataset('./benchmark_datasets/Lipophilicity_dataset/Lipophilicity_test.csv',
                            cols_to_read=[0], pad=False, tokens=train_dataset.tokens, augment=False)

assert train_dataset.tokens == val_dataset.tokens

train_dataset.target = train_dataset.target.reshape(-1, 1)
val_dataset.target = val_dataset.target.reshape(-1, 1)

use_cuda = True

model = GenerativeRNN

model_params = {
    'use_cuda': use_cuda,
    'task': 'classification',
    'random_seed': 5,
    'use_clip_grad': True,
    'max_grad_norm': 10.0,
    'batch_size': 1,
    'num_epochs': 100,
    'logdir': '/home/mpopova/Work/OpenChem/logs/generative_log',
    'print_every': 1,
    'save_every': 5,
    'train_data_layer': train_dataset,
    'val_data_layer': val_dataset,
    'eval_metrics': None,
    'criterion': nn.CrossEntropyLoss,
    'optimizer': RMSprop,
    'optimizer_params': {
        'lr': 0.005,
        #'weight_decay': 1e-4
        },
    'lr_scheduler': ExponentialLR,
    'lr_scheduler_params': {
        'gamma': 0.97
    },
    'embedding': Embedding,
    'embedding_params': {
        'num_embeddings': train_dataset.num_tokens,
        'embedding_dim': 100,
        'padding_idx': train_dataset.tokens.index(' ')
    },
    'encoder': RNNEncoder,
    'encoder_params': {
        'input_size': 100,
        'layer': "GRU",
        'encoder_dim': 100,
        'n_layers': 2,
        'dropout': 0.8,
        'is_bidirectional': False
    },
    'has_stack': True,
    'stack_params': {
        'stack_width': 100,
        'stack_depth': 200
    },
    'mlp': OpenChemMLP,
    'mlp_params': {
        'input_size': 100,
        'n_layers': 2,
        'hidden_size': [100, train_dataset.num_tokens],
        'activation': F.relu,
        'dropout': 0.5
    }
}

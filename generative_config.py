from openchem.models.GenerativeRNN import GenerativeRNN
from openchem.modules.embeddings.basic_embedding import Embedding
from openchem.modules.encoders.rnn_encoder import RNNEncoder
from openchem.modules.mlp.openchem_mlp import OpenChemMLP
from openchem.data.smiles_data_layer import SmilesDataset

import torch.nn as nn

from torch.optim import RMSprop, Adadelta
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
import pickle

train_dataset = SmilesDataset('./benchmark_datasets/Lipophilicity_dataset/Lipophilicity_test.csv', delimiter=',',
                              cols_to_read=[0], pad=True, tokenize=True, augment=False)
#train_dataset = pickle.load(open('./benchmark_datasets/chembl/chembl.pkl', 'rb')) 
print('Dataset loaded')
train_dataset.tokens += '<>'
train_dataset.num_tokens = len(train_dataset.tokens)

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
    'val_data_layer': None,
    'eval_metrics': None,
    'criterion': nn.CrossEntropyLoss(ignore_index=train_dataset.tokens.index(' ')),
    'optimizer': RMSprop,
    'optimizer_params': {
        'lr': 0.0001,
        },
    'lr_scheduler': ExponentialLR,
    'lr_scheduler_params': {
        'gamma': 1.0
    },
    'embedding': Embedding,
    'embedding_params': {
        'num_embeddings': train_dataset.num_tokens,
        'embedding_dim': 128,
        'padding_idx': train_dataset.tokens.index(' ')
    },
    'encoder': RNNEncoder,
    'encoder_params': {
        'input_size': 128,
        'layer': "GRU",
        'encoder_dim': 256,
        'n_layers': 1,
        'dropout': 0.0,
        'is_bidirectional': False
    },
    'has_stack': False,
    #'stack_params': {
    #    'in_features': 100,
    #    'stack_width': 100,
    #    'stack_depth': 200
    #},
    'mlp': OpenChemMLP,
    'mlp_params': {
        'input_size': 256,
        'n_layers': 2,
        'hidden_size': [128, train_dataset.num_tokens],
        'activation': F.relu,
        'dropout': 0.8
    }
}

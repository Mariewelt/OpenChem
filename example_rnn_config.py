from models.Smiles2Label import Smiles2Label
from modules.embeddings.basic_embedding import Embedding
from modules.encoders.rnn_encoder import RNNEncoder
from modules.encoders.cnn_encoder import CNNEncoder
from modules.mlp.openchem_mlp import OpenChemMLP
from data.smiles_data_layer import SmilesDataset

import torch.nn as nn
from torch.optim import RMSprop, Adam
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

train_dataset = SmilesDataset('./benchmark_datasets/HIV_dataset/HIV_train.csv',
                              cols_to_read=[0, 1])
val_dataset = SmilesDataset('./benchmark_datasets/HIV_dataset/HIV_test.csv',
                            cols_to_read=[0, 1], tokens=train_dataset.tokens)
assert train_dataset.tokens == val_dataset.tokens

use_cuda = True

model = Smiles2Label

model_params = {
    'use_cuda': use_cuda,
    'task': 'classification',
    'random_seed': 5,
    'use_clip_grad': True,
    'max_grad_norm': 1.0,
    'batch_size': 128,
    'num_epochs': 200,
    'logdir': '/home/mpopova/Work/OpenChem/logs/rnn_log',
    'print_every': 1,
    'save_every': 5,
    'train_data_layer': train_dataset,
    'val_data_layer': val_dataset,
    'eval_metrics': roc_auc_score,
    'criterion': nn.CrossEntropyLoss(),
    'optimizer': RMSprop,
    'optimizer_params': {
        'lr': 0.0001,
	'weight_decay': 1e-5
    },
    'lr_scheduler': ExponentialLR,
    'lr_scheduler_params': {
        'gamma': 0.97
    },
    'embedding': Embedding,
    'embedding_params': {
        'num_embeddings': max(train_dataset.num_tokens, val_dataset.num_tokens),
        'embedding_dim': 128,
        'padding_idx': 0
    },
    'encoder': RNNEncoder,
    'encoder_params': {
        'input_size': 128,
        'layer': "LSTM",
        'encoder_dim': 64,
        'n_layers': 2,
        'dropout': 0.8,
        'is_bidirectional': False
    },
    'mlp': OpenChemMLP,
    'mlp_params': {
        'input_size': 64,
        'n_layers': 2,
        'hidden_size': [64, 2],
        'activations': [F.relu, F.relu],
        'dropout': 0.8
    }
}

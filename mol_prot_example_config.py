from models.MoleculeProtein2Label import MoleculeProtein2Label
from modules.embeddings.basic_embedding import Embedding
from modules.encoders.rnn_encoder import RNNEncoder
from modules.mlp.openchem_mlp import OpenChemMLP
from data.smiles_protein_data_layer import SmilesProteinDataset

import torch.nn as nn
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
from sklearn.metrics import f1_score

i = 2
train_dataset = SmilesProteinDataset('/home/mpopova/Work/data/cv' + str(i) + '_train.pkl',
                              tokenized=True)
val_dataset = SmilesProteinDataset('/home/mpopova/Work/data/test.pkl',
#val_dataset = SmilesProteinDataset('/home/mpopova/Work/data/cv' + str(i) + '_test.pkl',
                            tokenized=True)

use_cuda = True

model = MoleculeProtein2Label

model_params = {
    'use_cuda': use_cuda,
    'task': 'classification',
    'random_seed': 5,
    'use_clip_grad': True,
    'max_grad_norm': 10.0,
    'batch_size': 128,
    'num_epochs': 100,
    'logdir': '/home/mpopova/Work/OpenChem/logs/kinase_model_logs/cv' + str(i),
    'print_every': 1,
    'save_every': 5,
    'train_data_layer': train_dataset,
    'val_data_layer': val_dataset,
    'eval_metrics': f1_score,
    'criterion': nn.CrossEntropyLoss(),
    'optimizer': RMSprop,
    'optimizer_params': {
        'lr': 0.001
    },
    'lr_scheduler': ExponentialLR,
    'lr_scheduler_params': {
        'gamma': 0.97
    },
    'mol_embedding': Embedding,
    'mol_embedding_params': {
        'num_embeddings': train_dataset.mol_num_tokens,
        'embedding_dim': 200,
        'padding_idx': train_dataset.mol_tokens.index(' ')
    },
    'prot_embedding': Embedding,
    'prot_embedding_params': {
        'num_embeddings': train_dataset.prot_num_tokens,
        'embedding_dim': 200,
        'padding_idx': train_dataset.prot_tokens.index(' ')
    },
    'mol_encoder': RNNEncoder,
    'mol_encoder_params': {
        'input_size': 200,
        'layer': "LSTM",
        'encoder_dim': 100,
        'n_layers': 2,
        'dropout': 0.8,
        'is_bidirectional': False
    },
    'prot_encoder': RNNEncoder,
    'prot_encoder_params': {
        'input_size': 200,
        'layer': "LSTM",
        'encoder_dim': 100,
        'n_layers': 2,
        'dropout': 0.8,
        'is_bidirectional': False
    },
    'merge': 'sum',
    'mlp': OpenChemMLP,
    'mlp_params': {
        'input_size': 100,
        'n_layers': 2,
        'hidden_size': [200, 2],
        'activation': F.relu,
        'dropout': 0.8
    }
}

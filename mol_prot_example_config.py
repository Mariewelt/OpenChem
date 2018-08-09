from openchem.models.MoleculeProtein2Label import MoleculeProtein2Label
from openchem.modules.embeddings.basic_embedding import Embedding
from openchem.modules.encoders.rnn_encoder import RNNEncoder
from openchem.modules.mlp.openchem_mlp import OpenChemMLP
from openchem.data.smiles_protein_data_layer import SmilesProteinDataset

import torch.nn as nn
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
from sklearn.metrics import f1_score

i = 4
train_dataset = SmilesProteinDataset('/result/train.txt', cols_to_read=[0, 1, 2],
                              tokenized=False)
val_dataset = SmilesProteinDataset('/result/test.txt', cols_to_read=[0, 1, 2],
                                   mol_tokens=train_dataset.mol_tokens, prot_tokens=train_dataset.prot_tokens, tokenized=False)
use_cuda = True

model = MoleculeProtein2Label

model_params = {
    'use_cuda': use_cuda,
    'task': 'classification',
    'random_seed': 5,
    'use_clip_grad': True,
    'max_grad_norm': 10.0,
    'batch_size': 128,
    'num_epochs': 150,
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
        'embedding_dim': 256,
        'padding_idx': train_dataset.mol_tokens.index(' ')
    },
    'prot_embedding': Embedding,
    'prot_embedding_params': {
        'num_embeddings': train_dataset.prot_num_tokens,
        'embedding_dim': 256,
        'padding_idx': train_dataset.prot_tokens.index(' ')
    },
    'mol_encoder': RNNEncoder,
    'mol_encoder_params': {
        'input_size': 256,
        'layer': "LSTM",
        'encoder_dim': 128,
        'n_layers': 2,
        'dropout': 0.8,
        'is_bidirectional': False
    },
    'prot_encoder': RNNEncoder,
    'prot_encoder_params': {
        'input_size': 256,
        'layer': "LSTM",
        'encoder_dim': 128,
        'n_layers': 2,
        'dropout': 0.8,
        'is_bidirectional': False
    },
    'merge': 'mul',
    'mlp': OpenChemMLP,
    'mlp_params': {
        'input_size': 128,
        'n_layers': 2,
        'hidden_size': [256, 2],
        'activation': F.relu,
        'dropout': 0.8
    }
}

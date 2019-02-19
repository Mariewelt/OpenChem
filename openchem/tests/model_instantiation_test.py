"""
Unit test and regression test for OpenChem models
"""

import torch
import pytest
import sys
from openchem.models.openchem_model import OpenChemModel
from openchem.models.Graph2Label import Graph2Label
from openchem.models.Smiles2Label import Smiles2Label
from openchem.models.MoleculeProtein2Label import MoleculeProtein2Label
from openchem.models.GenerativeRNN import GenerativeRNN
from openchem.modules.embeddings.openchem_embedding import OpenChemEmbedding
from openchem.modules.encoders.rnn_encoder import RNNEncoder
from openchem.modules.encoders.gcn_encoder import GraphCNNEncoder
from openchem.modules.mlp.openchem_mlp import OpenChemMLP


def test_openchem_model_instantiation():
    sample_config = {
        'task': 'classification',
        'batch_size': 16,
        'num_epochs': 1,
        'train_data_layer': None,
        'val_data_layer': None,
        'use_cuda': True,
        'eval_metrics': None,
        'logdir': '/logs',
        'world_size': 1,
        'use_clip_grad': False,
        'random_seed': 42,
        'print_every': 10,
        'plot_every': 10,
        'save_every': 10
    }
    new_model = OpenChemModel(sample_config)

def test_graph2label_model_instantiation():
    sample_config = {
        'task': 'classification',
        'batch_size': 16,
        'num_epochs': 1,
        'train_data_layer': None,
        'val_data_layer': None,
        'use_cuda': True,
        'eval_metrics': None,
        'logdir': '/logs',
        'world_size': 1,
        'use_clip_grad': False,
        'random_seed': 42,
        'print_every': 10,
        'plot_every': 10,
        'save_every': 10,
        'optimizer': torch.optim.Adam,
        'optimizer_params': {
            'lr': 0.01
        },
        'lr_scheduler': torch.optim.lr_scheduler.StepLR,
        'lr_scheduler_params': {
            'step_size': 10,
            'gamma': 0.8,
        },
        'encoder': GraphCNNEncoder,
        'encoder_params': {
            'input_size': 12,
            'hidden_size': [12],
            'encoder_dim': 12,
            'n_layers': 1,
        },
        'mlp': OpenChemMLP,
        'mlp_params': {
            'hidden_size': [12],
            'input_size': 12,
            'n_layers': 1,
            'activation': torch.tanh
        }
    }
    new_model = Graph2Label(sample_config)


def test_smiles2label_model_instantiation():
    sample_config = {
        'task': 'classification',
        'batch_size': 16,
        'num_epochs': 1,
        'train_data_layer': None,
        'val_data_layer': None,
        'use_cuda': True,
        'eval_metrics': None,
        'logdir': '/logs',
        'world_size': 1,
        'use_clip_grad': False,
        'random_seed': 42,
        'print_every': 10,
        'plot_every': 10,
        'save_every': 10,
        'optimizer': torch.optim.Adam,
        'optimizer_params': {
            'lr': 0.01
        },
        'lr_scheduler': torch.optim.lr_scheduler.StepLR,
        'lr_scheduler_params': {
            'step_size': 10,
            'gamma': 0.8,
        },
        'embedding': OpenChemEmbedding,
        'embedding_params': {
            'num_embeddings': 10,
            'embedding_dim': 12,
            #'padding_idx': None
        },
        'encoder': RNNEncoder,
        'encoder_params': {
            'input_size': 12,
            'layer': 'LSTM',
            'encoder_dim': 12,
            'n_layers': 1,
            'is_bidirectional': False
        },
        'mlp': OpenChemMLP,
        'mlp_params': {
            'hidden_size': [12],
            'input_size': 12,
            'n_layers': 1,
            'activation': torch.tanh
        }
    }
    new_model = Smiles2Label(sample_config)


def test_molecule_protein2label_model_instantiation():
    sample_config = {
        'task': 'classification',
        'batch_size': 16,
        'num_epochs': 1,
        'train_data_layer': None,
        'val_data_layer': None,
        'use_cuda': True,
        'eval_metrics': None,
        'logdir': '/logs',
        'world_size': 1,
        'use_clip_grad': False,
        'random_seed': 42,
        'print_every': 10,
        'plot_every': 10,
        'save_every': 10,
        'optimizer': torch.optim.Adam,
        'optimizer_params': {
            'lr': 0.01
        },
        'lr_scheduler': torch.optim.lr_scheduler.StepLR,
        'lr_scheduler_params': {
            'step_size': 10,
            'gamma': 0.8,
        },
        'mol_embedding': OpenChemEmbedding,
        'mol_embedding_params': {
            'num_embeddings': 10,
            'embedding_dim': 12,
            #'padding_idx': None
        },
        'mol_encoder': RNNEncoder,
        'mol_encoder_params': {
            'input_size': 12,
            'layer': 'LSTM',
            'encoder_dim': 12,
            'n_layers': 1,
            'is_bidirectional': False
        },
        'prot_embedding': OpenChemEmbedding,
        'prot_embedding_params': {
            'num_embeddings': 10,
            'embedding_dim': 12,
            # 'padding_idx': None
        },
        'prot_encoder': RNNEncoder,
        'prot_encoder_params': {
            'input_size': 12,
            'layer': 'LSTM',
            'encoder_dim': 12,
            'n_layers': 1,
            'is_bidirectional': False
        },
        'merge': 'concat',
        'mlp': OpenChemMLP,
        'mlp_params': {
            'hidden_size': [12],
            'input_size': 12,
            'n_layers': 1,
            'activation': torch.tanh
        }
    }
    new_model = MoleculeProtein2Label(sample_config)


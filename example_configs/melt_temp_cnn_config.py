from openchem.models.Smiles2Label import Smiles2Label
from openchem.modules.embeddings.basic_embedding import Embedding
from openchem.modules.encoders.cnn_encoder import CNNEncoder
from openchem.modules.mlp.openchem_mlp import OpenChemMLP
from openchem.data.smiles_data_layer import SmilesDataset
from openchem.criterion.multitask_loss import MultitaskLoss
from sklearn.metrics import r2_score, mean_squared_error
from openchem.utils.utils import identity
import torch
import torch.nn as nn

import numpy as np

from torch.optim import RMSprop, Adam
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import torch.nn.functional as F

from openchem.data.utils import read_smiles_property_file
data = read_smiles_property_file('benchmark_datasets/melt_temp/melting_data.txt',
                                 cols_to_read=[0, 1], delimiter='\t',
                                 keep_header=False)
smiles = data[0][1:]
labels = np.array(data[1][1:], dtype='float').reshape(-1)

from openchem.data.utils import get_tokens
tokens, _, _ = get_tokens(smiles)
tokens = tokens + ' '

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(smiles, labels,
                                                    test_size=0.2,
                                                    random_state=42)

train_mean = np.mean(y_train)
train_std = np.std(y_train)
print("Mean Tmelt in training data: ", train_mean)
print("Standard deviation of Tmelt in training data: ", train_std)
print("Min value of Tmelt in training data: ", np.min(y_train))
print("Max value of Tmelt in training data: ", np.max(y_train))
y_train = (y_train - train_mean) / train_std
y_test = (y_test - train_mean) / train_std

def rmse_tmelt(target, predicted, std=train_std):
    mse = mean_squared_error(target, predicted)
    rmse = np.sqrt(mse) * std
    return rmse

from openchem.data.utils import save_smiles_property_file
save_smiles_property_file('./benchmark_datasets/melt_temp/train.smi',
                          X_train,
                          y_train.reshape(-1, 1))

save_smiles_property_file('./benchmark_datasets/melt_temp/test.smi',
                          X_test,
                          y_test.reshape(-1, 1))

from openchem.data.smiles_data_layer import SmilesDataset
train_dataset = SmilesDataset('./benchmark_datasets/melt_temp/train.smi',
                              delimiter=',', cols_to_read=[0, 1],
                              tokens=tokens, augment=True)
test_dataset = SmilesDataset('./benchmark_datasets/melt_temp/test.smi',
                            delimiter=',', cols_to_read=[0, 1],
                            tokens=tokens)

model = Smiles2Label

model_params = {
    'use_cuda': True,
    'task': 'regression',
    'random_seed': 5,
    'use_clip_grad': True,
    'max_grad_norm': 10.0,
    'batch_size': 256,
    'num_epochs': 101,
    'logdir': './logs/tmelt_cnn_log',
    'print_every': 1,
    'save_every': 5,
    'train_data_layer': train_dataset,
    'val_data_layer': test_dataset,
     'eval_metrics': rmse_tmelt,
    'criterion': nn.MSELoss(),
    'optimizer': RMSprop,
    'optimizer_params': {
        'lr': 0.001,
        },
    'lr_scheduler': StepLR,
    'lr_scheduler_params': {
        'step_size': 10,
        'gamma': 0.8
    },
    'embedding': Embedding,
    'embedding_params': {
        'num_embeddings': train_dataset.num_tokens,
        'embedding_dim': 128,
        'padding_idx': train_dataset.tokens.index(' ')
    },
    'encoder': CNNEncoder,
    'encoder_params': {
        'input_size': 128,
        'encoder_dim': 256,
        'kernel_sizes': [15, 12, 9, 7, 3],
        'dropout': 0.5,
        'pooling': 'mean'
    },
    'mlp': OpenChemMLP,
    'mlp_params': {
        'input_size': 256,
        'n_layers': 2,
        'hidden_size': [128, 1],
        'activation': [F.relu, identity],
        'dropout': 0.0
    }
}

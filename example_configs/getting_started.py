# This config file provides an example of building a multi-layer perceptron for prediction of logP values
import numpy as np

from openchem.models.MLP2Label import MLP2Label
from openchem.data.feature_data_layer import FeatureDataset
from openchem.modules.mlp.openchem_mlp import OpenChemMLP
from openchem.data.utils import get_fp
from openchem.utils.utils import identity
from openchem.data.utils import read_smiles_property_file
from openchem.data.utils import save_smiles_property_file

import torch.nn as nn
from torch.optim import RMSprop, SGD, Adam
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import torch.nn.functional as F

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


data = read_smiles_property_file('./benchmark_datasets/logp_dataset/logP_labels.csv',
                                 cols_to_read=[1, 2],
                                 keep_header=False)

smiles = data[0]
labels = np.array(data[1:])
labels = labels.T

X_train, X_test, y_train, y_test = train_test_split(smiles, labels, test_size=0.2,
                                                    random_state=42)

save_smiles_property_file('./benchmark_datasets/logp_dataset/train.smi', X_train, y_train)
save_smiles_property_file('./benchmark_datasets/logp_dataset/test.smi', X_test, y_test)

train_dataset = FeatureDataset(filename='./benchmark_datasets/logp_dataset/train.smi',
                               delimiter=',', cols_to_read=[0, 1],
                               get_features=get_fp, get_features_args={"n_bits": 2048})
test_dataset = FeatureDataset(filename='./benchmark_datasets/logp_dataset/test.smi',
                              delimiter=',', cols_to_read=[0, 1],
                              get_features=get_fp, get_features_args={"n_bits": 2048})
predict_dataset = FeatureDataset(filename='./benchmark_datasets/logp_dataset/test.smi',
                                delimiter=',', cols_to_read=[0],
                                get_features=get_fp, get_features_args={"n_bits": 2048},
                                return_smiles=True)

model = MLP2Label

model_params = {
    'task': 'regression',
    'random_seed': 42,
    'batch_size': 256,
    'num_epochs': 101,
    'logdir': 'logs/logp_mlp_logs',
    'print_every': 20,
    'save_every': 5,
    'train_data_layer': train_dataset,
    'val_data_layer': test_dataset,
    'predict_data_layer': predict_dataset,
    'eval_metrics': r2_score,
    'criterion': nn.MSELoss(),
    'optimizer': Adam,
    'optimizer_params': {
        'lr': 0.001,
    },
    'lr_scheduler': StepLR,
    'lr_scheduler_params': {
        'step_size': 15,
        'gamma': 0.9
    },
    'mlp': OpenChemMLP,
    'mlp_params': {
        'input_size': 2048,
        'n_layers': 4,
        'hidden_size': [1024, 512, 128, 1],
        'dropout': 0.5,
        'activation': [F.relu, F.relu, F.relu, identity]
    }
}

from openchem.models.Graph2Label import Graph2Label
from openchem.modules.encoders.gcn_encoder import GraphCNNEncoder
from openchem.modules.mlp.openchem_mlp import OpenChemMLP
from openchem.data.graph_data_layer import GraphDataset

from openchem.utils.graph import Attribute
from openchem.utils.utils import identity

import torch.nn as nn
from torch.optim import RMSprop, SGD, Adam
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_squared_error

import numpy as np
import pandas as pd

import copy
import pickle

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

from openchem.data.utils import save_smiles_property_file
save_smiles_property_file('./benchmark_datasets/melt_temp/train.smi',
                          X_train,
                          y_train.reshape(-1, 1))

save_smiles_property_file('./benchmark_datasets/melt_temp/test.smi',
                          X_test,
                          y_test.reshape(-1, 1))

def get_atomic_attributes(atom):
    attr_dict = {}
  
    atomic_num = atom.GetAtomicNum()
    atomic_mapping = {5: 0, 7: 1, 6: 2, 8: 3, 9: 4, 15: 5, 16: 6, 17: 7, 35: 8,
                      53: 9}
    if atomic_num in atomic_mapping.keys():
        attr_dict['atom_element'] = atomic_mapping[atomic_num]
    else:
        attr_dict['atom_element'] = 10
    attr_dict['valence'] = atom.GetTotalValence()
    attr_dict['charge'] = atom.GetFormalCharge()
    attr_dict['hybridization'] = atom.GetHybridization().real
    attr_dict['aromatic'] = int(atom.GetIsAromatic())
    return attr_dict


node_attributes = {}
node_attributes['valence'] = Attribute('node', 'valence', one_hot=True,
                                       values=[1, 2, 3, 4, 5, 6, 7])

node_attributes['charge'] = Attribute('node', 'charge', one_hot=True,
                                      values=[-1, 0, 1, 2, 3, 4])

node_attributes['hybridization'] = Attribute('node', 'hybridization',
                                             one_hot=True,
                                             values=[0, 1, 2, 3, 4, 5, 6, 7])

node_attributes['aromatic'] = Attribute('node', 'aromatic', one_hot=True,
                                        values=[0, 1])

node_attributes['atom_element'] = Attribute('node', 'atom_element',
                                            one_hot=True,
                                            values=list(range(11)))

train_dataset = GraphDataset(get_atomic_attributes, node_attributes,
                             './benchmark_datasets/melt_temp/train.smi',
                             delimiter=',', cols_to_read=[0, 1])

test_dataset = GraphDataset(get_atomic_attributes, node_attributes,
                             './benchmark_datasets/melt_temp/test.smi',
                             delimiter=',', cols_to_read=[0, 1])

predict_dataset = GraphDataset(get_atomic_attributes, node_attributes,
                               './benchmark_datasets/melt_temp/test.smi',
                               delimiter=',', cols_to_read=[0],
                               return_smiles=True)


sample = train_dataset[0]
num_features = sample['node_feature_matrix'].shape[1]

model = Graph2Label

model_params = {
    'random_seed': 42,
    'task': 'regression',
    'data_layer': GraphDataset,
    'use_clip_grad': False,
    'batch_size': 512,
    'num_epochs': 31,
    'logdir': 'logs/melt_temp_gcn_log',
    'print_every': 1,
    'save_every': 1,
    'train_data_layer': train_dataset,
    'val_data_layer': test_dataset,
    'predict_data_layer': predict_dataset,
    'eval_metrics': r2_score,
    'criterion': nn.MSELoss(),
    'optimizer': Adam,
    'optimizer_params': {
        'lr': 0.01,
    },
    'lr_scheduler': StepLR,
    'lr_scheduler_params': {
        'step_size': 15,
        'gamma': 0.99
    },
    'encoder': GraphCNNEncoder,
    'encoder_params': {
        'input_size': num_features,
        'encoder_dim': 128,
        'n_layers': 3,
        'hidden_size': [128, 128, 128],
    },
    'mlp': OpenChemMLP,
    'mlp_params': {
        'input_size': 128,
        'n_layers': 2,
        'hidden_size': [128, 1],
        'activation': [F.relu, identity]
    }
}

new_model_params = model_params.copy()
new_model_params['train_data_layer'] = None
new_model_params['val_data_layer'] = None
new_model_params['world_size'] = 1
new_model_params['use_cuda'] = True

#import pickle
#pickle.dump(new_model_params, open(model_params['logdir'] + '/params.pkl', 'wb'))

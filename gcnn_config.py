from models.Graph2Label import Graph2Label
from modules.encoders.gcn_encoder import GraphCNNEncoder
from modules.mlp.openchem_mlp import OpenChemMLP
from data.graph_data_layer import GraphDataset

from utils.graph import Attribute

import torch.nn as nn
from torch.optim import RMSprop, SGD, Adam
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_squared_error

import pandas as pd

import copy
import pickle

ref_data = pd.read_csv('/home/mpopova/Work/OpenChem/benchmark_datasets/atomic_data.csv',
                       index_col=0)


def get_atomic_attributes(atom):
    attr_dict = {}
    # periodic table
    atomic_num = atom.GetAtomicNum()
    atomic_mapping = {5: 0, 7: 1, 6: 2, 8: 3, 9: 4, 15: 5, 16: 6, 17: 7, 35: 8,
                      53: 9}
    if atomic_num in atomic_mapping.keys():
        attr_dict['atom_element'] = atomic_mapping[atomic_num]
    else:
        attr_dict['atom_element'] = 10
    # attr_dict['group'] =  ref_data.loc[atom.GetAtomicNum()]['Group']
    # attr_dict['period'] =  ref_data.loc[atom.GetAtomicNum()]['Period']
    #attr_dict['mass'] = ref_data.loc[atom.GetAtomicNum()]['Mass']
    #attr_dict['molar_volume'] = ref_data.loc[atom.GetAtomicNum()]['MolarVol']
    #attr_dict['density'] = ref_data.loc[atom.GetAtomicNum()]['Density']
    # mol structure
    attr_dict['valence'] = atom.GetTotalValence()
    attr_dict['charge'] = atom.GetFormalCharge()
    attr_dict['hybridization'] = atom.GetHybridization().real
    attr_dict['aromatic'] = int(atom.GetIsAromatic())
    # ref expt data
    #attr_dict['z_eff'] = ref_data.loc[atom.GetAtomicNum()]['Zeff']
    #attr_dict['radii_abs'] = ref_data.loc[atom.GetAtomicNum()]['RadiiAbs']
    #attr_dict['radii_cov'] = ref_data.loc[atom.GetAtomicNum()]['RadiiCordero08']

    #attr_dict['IP1'] = ref_data.loc[atom.GetAtomicNum()]['IP1']
    #attr_dict['IP2'] = ref_data.loc[atom.GetAtomicNum()]['IP2']
    #attr_dict['IP3'] = ref_data.loc[atom.GetAtomicNum()]['IP3']
    #attr_dict['EA'] = ref_data.loc[atom.GetAtomicNum()]['EA']

    #attr_dict['polarisability'] = ref_data.loc[atom.GetAtomicNum()][
    #    'DipolePolaris']
    #attr_dict['heat_capacity'] = ref_data.loc[atom.GetAtomicNum()]['HeatCap']
    #attr_dict['heat_fusion'] = ref_data.loc[atom.GetAtomicNum()]['dHFus']
    #attr_dict['heat_atom'] = ref_data.loc[atom.GetAtomicNum()]['dHAtom']
    #attr_dict['thermal_cond'] = ref_data.loc[atom.GetAtomicNum()]['ThermalCond']
    return attr_dict


node_attributes = {}
#node_attributes['mass'] = Attribute('node', 'mass', one_hot=False)
#node_attributes['molar_volume'] = Attribute('node', 'molar_volume',
#                                            one_hot=False)
#node_attributes['density'] = Attribute('node', 'density', one_hot=False)
# mol structure
node_attributes['valence'] = Attribute('node', 'valence', one_hot=True, values=[1, 2, 3, 4, 5, 6])
node_attributes['charge'] = Attribute('node', 'charge', one_hot=True, values=[0, 1, 2, 3, 4])
node_attributes['hybridization'] = Attribute('node', 'hybridization',
                                             one_hot=True, values=[0, 1, 2, 3, 4, 5, 6, 7])
node_attributes['aromatic'] = Attribute('node', 'aromatic', one_hot=True,
                                        values=[0, 1])
# ref expt data
#node_attributes['z_eff'] = Attribute('node', 'z_eff', one_hot=False)
#node_attributes['radii_abs'] = Attribute('node', 'radii_abs', one_hot=False)
#node_attributes['radii_cov'] = Attribute('node', 'radii_cov', one_hot=False)

#node_attributes['IP1'] = Attribute('node', 'IP1', one_hot=False)
#node_attributes['IP2'] = Attribute('node', 'IP2', one_hot=False)
#node_attributes['IP3'] = Attribute('node', 'IP3', one_hot=False)
#node_attributes['EA'] = Attribute('node', 'EA', one_hot=False)

#node_attributes['polarisability'] = Attribute('node', 'polarisability',
#                                              one_hot=False)
#node_attributes['heat_capacity'] = Attribute('node', 'heat_capacity',
#                                             one_hot=False)
#node_attributes['heat_fusion'] = Attribute('node', 'heat_fusion', one_hot=False)
#node_attributes['heat_atom'] = Attribute('node', 'heat_atom', one_hot=False)
#node_attributes['thermal_cond'] = Attribute('node', 'thermal_cond',
#                                            one_hot=False)
node_attributes['atom_element'] = Attribute('node', 'atom_element',
                                            one_hot=True,
                                            values=list(range(11)))
#train_dataset = GraphDataset(get_atomic_attributes, node_attributes,
#                             './benchmark_datasets/Lipophilicity_dataset/Lipophilicity_train.csv',
#                             delimiter=',', cols_to_read=[0, 1])
#val_dataset = GraphDataset(get_atomic_attributes, node_attributes,
#                             './benchmark_datasets/Lipophilicity_dataset/Lipophilicity_test.csv',
#                             delimiter=',', cols_to_read=[0, 1])

train_dataset = pickle.load(open('/home/mpopova/Work/OpenChem/benchmark_datasets/Lipophilicity_dataset/train.pkl', 'rb'))
val_dataset = pickle.load(open('/home/mpopova/Work/OpenChem/benchmark_datasets/Lipophilicity_dataset/test.pkl', 'rb'))

num_features = train_dataset.node_feature_matrix[0].shape[1]

train_dataset.target = train_dataset.target.reshape(-1, 1)
val_dataset.target = val_dataset.target.reshape(-1, 1)


model = Graph2Label

model_params = {
    'task': 'regression',
    'random_seed': 5,
    'use_clip_grad': False,
    'max_grad_norm': 10.0,
    'batch_size': 256,
    'num_epochs': 100,
    'logdir': '/home/mpopova/Work/OpenChem/logs/gcnn_logs',
    'print_every': 1,
    'save_every': 5,
    'train_data_layer': train_dataset,
    'val_data_layer': val_dataset,
    'eval_metrics': mean_squared_error,
    'criterion': nn.MSELoss(),
    'optimizer': Adam,
    'optimizer_params': {
        'lr': 0.001,
    },
    'lr_scheduler': ExponentialLR,
    'lr_scheduler_params': {
        'gamma': 1.0
    },
    'encoder': GraphCNNEncoder,
    'encoder_params': {
        'input_size': num_features,
        'encoder_dim': 128,
        'dropout': 0.0,
        'n_layers': 3,
        'hidden_size': [64, 128, 64],
    },
    'mlp': OpenChemMLP,
    'mlp_params': {
        'input_size': 128,
        'n_layers': 2,
        'hidden_size': [128, 1],
        'activation': F.relu,
        'dropout': 0.8
    }
}

# my_model = model(model_params)

# if use_cuda:
#     my_model = my_model.cuda()

# my_model.fit()
# my_model.evaluate()

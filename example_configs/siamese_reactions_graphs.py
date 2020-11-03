from openchem.models.SiameseModel import SiameseModel
from openchem.modules.embeddings.basic_embedding import Embedding
from openchem.modules.encoders.rnn_encoder import RNNEncoder
from openchem.modules.encoders.gcn_encoder import GraphCNNEncoder
from openchem.modules.mlp.openchem_mlp import OpenChemMLP, OpenChemMLPSimple
from openchem.data.siamese_data_layer import SiameseDataset
from openchem.utils.utils import identity

from openchem.utils.graph import Attribute
import torch
import torch.nn as nn

import numpy as np

from torch.optim import RMSprop, Adam
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score, f1_score

from openchem.data.utils import read_smiles_property_file
data = read_smiles_property_file('./benchmark_datasets/reactions/4_11_with_y2.csv',
                                 cols_to_read=[11, 12, 14],
                                 keep_header=False)
reactant1 = data[0]
reactant2 = data[1]
labels = np.array(data[2], dtype="float").reshape(-1, 1)

reactants = [reactant1[i] + " " + reactant2[i] for i in range(len(reactant2))]

from openchem.data.utils import get_tokens
tokens, _, _ = get_tokens(reactants)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(reactants, labels, test_size=0.2,
                                                    random_state=42)
y_mean = np.mean(y_train)
y_std = np.std(y_train)
y_train = (y_train - y_mean) / y_std
y_test = (y_test - y_mean) / y_std

from openchem.data.utils import save_smiles_property_file
save_smiles_property_file('./benchmark_datasets/reactions/train.smi', X_train, y_train, delimiter=" ")
save_smiles_property_file('./benchmark_datasets/reactions/test.smi', X_test, y_test, delimiter=" ")

from openchem.data.smiles_data_layer import SmilesDataset

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
node_attributes['valence'] = Attribute('node', 'valence', one_hot=True, values=[1, 2, 3, 4, 5, 6])
node_attributes['charge'] = Attribute('node', 'charge', one_hot=True, values=[-1, 0, 1, 2, 3, 4])
node_attributes['hybridization'] = Attribute('node', 'hybridization',
                                             one_hot=True, values=[0, 1, 2, 3, 4, 5, 6, 7])
node_attributes['aromatic'] = Attribute('node', 'aromatic', one_hot=True,
                                        values=[0, 1])
node_attributes['atom_element'] = Attribute('node', 'atom_element',
                                            one_hot=True,
                                            values=list(range(11)))

head1_arguments = {
    "get_atomic_attributes": get_atomic_attributes,
    "node_attributes": node_attributes,
    "delimiter": " "
}

head2_arguments = {
    "get_atomic_attributes": get_atomic_attributes,
    "node_attributes": node_attributes,
    "delimiter": " "
}

train_dataset = SiameseDataset('./benchmark_datasets/reactions/train.smi',
                               head1_type='graphs', head2_type='graphs', cols_to_read=[0, 1, 2],
                               head1_arguments=head1_arguments, head2_arguments=head2_arguments)


test_dataset = SiameseDataset('./benchmark_datasets/reactions/test.smi',
                               head1_type='graphs', head2_type='graphs', cols_to_read=[0, 1, 2],
                               head1_arguments=head1_arguments, head2_arguments=head2_arguments)

model = SiameseModel

model_params = {
    'use_cuda': True,
    'task': 'classification',
    'random_seed': 5,
    'use_clip_grad': True,
    'max_grad_norm': 10.0,
    'batch_size': 256,
    'num_epochs': 51,
    'logdir': './logs/reactions_graphs',
    'print_every': 1,
    'save_every': 1,
    'train_data_layer': train_dataset,
    'val_data_layer': test_dataset,
    'eval_metrics': f1_score,
    'criterion': nn.CrossEntropyLoss(),#nn.MSELoss(),
    'optimizer': Adam,
    'optimizer_params': {
        'lr': 0.001,
        },
    'lr_scheduler': StepLR,
    'lr_scheduler_params': {
        'step_size': 10,
        'gamma': 0.8
    },
    'head1_embedding': None,
    'head1_embedding_params': None,
    'head2_embedding': None,
    'head2_embedding_params': None,
    'head1_encoder': GraphCNNEncoder,
    'head1_encoder_params': {
        'input_size': train_dataset[0]['head1']["node_feature_matrix"].shape[1],
        'encoder_dim': 128,
        'n_layers': 5,
        'hidden_size': [128, 128, 128, 128, 128],
    },
    'head2_encoder': GraphCNNEncoder,
    'head2_encoder_params': {
        'input_size': train_dataset[0]['head2']["node_feature_matrix"].shape[1],
        'encoder_dim': 128,
        'n_layers': 5,
        'hidden_size': [128, 128, 128, 128, 128],
    },
    'merge': "mul",
    'mlp': OpenChemMLPSimple,
    'mlp_params': {
        'input_size': 128,
        'n_layers': 2,
        'hidden_size': [128, 2],
        'activation': [F.relu, nn.Softmax(dim=1)],
    }
}

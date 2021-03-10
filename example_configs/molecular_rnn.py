from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from openchem.utils.graph import Attribute
from openchem.data.graph_data_layer import BFSGraphDataset
from openchem.models.MolecularRNN import MolecularRNNModel
from openchem.modules.embeddings.basic_embedding import Embedding
from openchem.modules.mlp.openchem_mlp import OpenChemMLPSimple
from openchem.utils.utils import identity
from openchem.modules.gru_plain import GRUPlain
from openchem.utils.metrics import qed, sa_score
from openchem.data.utils import DummyDataset

def get_sa_score(target, smiles):
    return sa_score(smiles)


def get_qed(target, smiles):
    return qed(smiles)


max_atom_bonds = [4., 3., 2., 1., 5., 6., 1., 1., 1.]

max_prev_nodes = 12
# this in Carbon original id in the Periodic Table
original_start_node_label = 6

edge_relabel_map = {
    0.: 0, 
    1.: 1, 
    2.: 2, 
    3.: 3
}

node_relabel_map = {6.0: 0, 7.0: 1, 8.0: 2, 9.0: 3, 15.0: 4, 16.0: 5, 17.0: 6, 35.0: 7, 53.0: 8}

atom2number = {
    'H': 1,
    'Be': 4,
    'B': 5,
    'C': 6,
    'N': 7,
    'O': 8,
    'F': 9,
    'Si': 14,
    'P': 15,
    'S': 16,
    'Cl': 17,
    'As': 33,
    'Se': 34,
    'Br': 35,
    'I': 53
}
number2atom = {i: v for v, i in atom2number.items()}


def get_atomic_attributes(atom):
    atomic_num = atom.GetAtomicNum()
    attr_dict = dict(atom_element=atomic_num)
    return attr_dict


def get_edge_attributes(bond):
    attr_dict = dict()
    attr_dict['bond_type'] = bond.GetBondTypeAsDouble()
    return attr_dict


node_attributes = dict(atom_element=Attribute('node', 'atom_element', one_hot=False), )

edge_attributes = dict(bond_type=Attribute('edge', 'bond_type', one_hot=False))

restrict_min_atoms = 10
restrict_max_atoms = 50

import pickle

train_dataset = BFSGraphDataset(get_atomic_attributes,
                                node_attributes,
                                file_format='pickled',
                                cols_to_read=[0, 1],
                                filename='benchmark_datasets/chembl_full/full_chembl_cleaned.pkl',
                                get_bond_attributes=get_edge_attributes,
                                edge_attributes=edge_attributes,
                                delimiter=',',
                                random_order=True,
                                max_prev_nodes=max_prev_nodes,
                                original_start_node_label=original_start_node_label,
                                edge_relabel_map=edge_relabel_map,
                                node_relabel_map=node_relabel_map,
                                restrict_min_atoms=restrict_min_atoms,
                                restrict_max_atoms=restrict_max_atoms)

val_dataset = DummyDataset()

num_edge_classes = train_dataset.num_edge_classes
num_node_classes = train_dataset.num_node_classes
node_relabel_map = train_dataset.node_relabel_map
inverse_node_relabel_map = train_dataset.inverse_node_relabel_map
max_num_nodes = train_dataset.max_num_nodes
start_node_label = train_dataset.start_node_label
label2atom = [number2atom[int(v)] for i, v in sorted(inverse_node_relabel_map.items())]
edge2type = [t for e, t in sorted(train_dataset.inverse_edge_relabel_map.items())]

edge_embedding_dim = 16

if num_edge_classes > 2:
    node_rnn_input_size = edge_embedding_dim * max_prev_nodes
    node_embedding_dim = 128
else:
    node_rnn_input_size = max_prev_nodes
    node_embedding_dim = max_prev_nodes
if num_node_classes > 2:
    node_rnn_input_size += node_embedding_dim

class DummyCriterion(object):
    def __call__(self, inp, out):
        return inp

    def cuda(self):
        return self


model = MolecularRNNModel
model_params = {
    'task': 'graph_generation',
    'use_cuda': True,
    'random_seed': 0,
    'use_clip_grad': False,
    'batch_size': 512,
    'num_epochs': 251,
    'logdir': './logs/molecular_rnn_log',
    'print_every': 1,
    'save_every': 5,
    'train_data_layer': train_dataset,
    'val_data_layer': val_dataset,
    'criterion': DummyCriterion(),
    'eval_metrics': get_sa_score,
    'optimizer': Adam,
    'optimizer_params': {
        'lr': 0.003,
    },
    'lr_scheduler': MultiStepLR,
    'lr_scheduler_params': {
        'milestones': [10, 30, 40, 45, 48],
        'gamma': 0.3
    },
    'num_node_classes': num_node_classes,
    'num_edge_classes': num_edge_classes,
    'max_num_nodes': max_num_nodes,
    'start_node_label': start_node_label,
    'max_prev_nodes': max_prev_nodes,
    'label2atom': label2atom,
    'edge2type': edge2type,
    "restrict_min_atoms": 10,
    "restrict_max_atoms": 100,
    "max_atom_bonds": max_atom_bonds,
    'EdgeEmbedding': Embedding,
    'edge_embedding_params': {
        'num_embeddings': num_edge_classes, 
        'embedding_dim': edge_embedding_dim
    },
    'NodeEmbedding': Embedding,
    'node_embedding_params': {
        'num_embeddings': num_node_classes, 
        'embedding_dim': node_embedding_dim
    },
    'NodeMLP': OpenChemMLPSimple,
    'node_mlp_params': {
        'input_size': 128,
        'n_layers': 2,
        'hidden_size': [128, num_node_classes],
        'activation': [nn.ReLU(inplace=True), identity],
        'init': "xavier_uniform"
     },
    'NodeRNN': GRUPlain,
    'node_rnn_params': {
        'input_size': node_rnn_input_size,
        'embedding_size': 128,
        'hidden_size': 256,
        'num_layers': 4,
        'has_input': True,
        'has_output': True,
        'has_output_nonlin': True,
        'output_size': 128  
    },
    'EdgeRNN': GRUPlain,
    'edge_rnn_params': {
        'input_size': edge_embedding_dim if num_edge_classes > 2 else 1,
        'embedding_size': 64,
        'hidden_size': 128,
        'num_layers': 4,
        'has_input': True,
        'has_output': True,
        'output_size': num_edge_classes if num_edge_classes > 2 else 1
    }
}


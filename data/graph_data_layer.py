# TODO: variable length batching

import numpy as np


from utils.graph import Graph

from rdkit import Chem

from torch.utils.data import Dataset
from data.utils import read_smiles_property_file, sanitize_smiles


class GraphDataset(Dataset):
    def __init__(self, get_atomic_attributes, node_attributes, filename,
                 cols_to_read, delimiter=','):
        super(GraphDataset, self).__init__()
        data_set = read_smiles_property_file(filename, cols_to_read,
                                                 delimiter)
        data = data_set[0]
        target = data_set[1:]
        clean_smiles, clean_idx = sanitize_smiles(data)
        target = np.array(target).T
        max_size = 0
        for sm in clean_smiles:
            mol = Chem.MolFromSmiles(sm)
            if mol.GetNumAtoms() > max_size:
                max_size = mol.GetNumAtoms()
        self.target = target[clean_idx, :]
        self.graphs = []
        self.node_feature_matrix = []
        self.adj_matrix = []
        for sm in clean_smiles:
            graph = Graph(sm, max_size, get_atomic_attributes)
            self.node_feature_matrix.append(
                graph.get_node_feature_matrix(node_attributes, max_size))
            self.adj_matrix.append(graph.adj_matrix)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        sample = {'adj_matrix': self.adj_matrix[index].astype('float32'),
                  'node_feature_matrix': self.node_feature_matrix[index].astype('float32'),
                  'labels': self.target[index].astype('float32')}
        return sample


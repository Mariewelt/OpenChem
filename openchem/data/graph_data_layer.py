# TODO: variable length batching

import numpy as np
import networkx as nx

from openchem.utils.graph import Graph

from rdkit import Chem

from torch.utils.data import Dataset
from openchem.data.utils import read_smiles_property_file, sanitize_smiles

from .graph_utils import bfs_seq, encode_adj


class GraphDataset(Dataset):
    def __init__(self, get_atomic_attributes, node_attributes, filename,
                 cols_to_read, delimiter=',', get_bond_attributes=None,
                 edge_attributes=None, **kwargs):
        super(GraphDataset, self).__init__()
        assert (get_bond_attributes is None) == (edge_attributes is None)
        data_set = read_smiles_property_file(filename, cols_to_read,
                                             delimiter)
        data = data_set[0]
        target = data_set[1:]
        clean_smiles, clean_idx = sanitize_smiles(data)
        target = np.array(target).T
        max_size = 0
        self.num_atoms_all = []
        for sm in clean_smiles:
            mol = Chem.MolFromSmiles(sm)
            num_atoms = mol.GetNumAtoms()
            max_size = max(max_size, num_atoms)
            self.num_atoms_all.append(num_atoms)
        self.target = target[clean_idx, :]
        self.graphs = []
        self.node_feature_matrix = []
        self.adj_matrix = []
        for sm in clean_smiles:
            graph = Graph(sm, max_size, get_atomic_attributes,
                          get_bond_attributes)
            self.node_feature_matrix.append(
                graph.get_node_feature_matrix(node_attributes, max_size))
            if get_bond_attributes is None:
                self.adj_matrix.append(graph.adj_matrix)
            else:
                self.adj_matrix.append(
                    graph.get_edge_attr_adj_matrix(edge_attributes, max_size)
                )
        self.num_features = self.node_feature_matrix[0].shape[1]

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        sample = {'adj_matrix': self.adj_matrix[index].astype('float32'),
                  'node_feature_matrix':
                      self.node_feature_matrix[index].astype('float32'),
                  'labels': self.target[index].astype('float32')}
        return sample


class BFSGraphDataset(GraphDataset):
    def __init__(self, *args, **kwargs):
        super(BFSGraphDataset, self).__init__(*args, **kwargs)
        self.random_order = kwargs["random_order"]
        self.max_prev_nodes = kwargs["max_prev_nodes"]
        self.num_edge_classes = kwargs
        self.max_num_nodes = max(self.num_atoms_all)
        original_start_node_label = kwargs.get(
            "original_start_node_label", None)

        if "node_relabel_map" not in kwargs:
            # define relabelling from Periodic Table numbers to {0, 1, ...}
            original_labels = np.concatenate(
                [n.flatten() for n in self.node_feature_matrix],
                axis=0
            )
            # remove paddings
            original_labels = original_labels[original_labels != 0]
            unique_labels = np.unique(original_labels)
            self.node_relabel_map = {
                v: i for i, v in enumerate(unique_labels)
            }
        else:
            self.node_relabel_map = kwargs["node_relabel_map"]
        self.inverse_node_relabel_map = {i: v for v, i in
                                         self.node_relabel_map.items()}

        if original_start_node_label is not None:
            self.start_node_label = \
                self.node_relabel_map[original_start_node_label]
        else:
            self.start_node_label = None

        if "edge_relabel_map" not in kwargs:
            raise NotImplementedError()
        else:
            self.edge_relabel_map = kwargs["edge_relabel_map"]
        self.inverse_edge_relabel_map = {i: v for v, i in
                                         self.edge_relabel_map.items()}

        self.num_node_classes = len(self.inverse_node_relabel_map)
        self.num_edge_classes = len(self.inverse_edge_relabel_map)

    def __getitem__(self, index):
        num_nodes = self.num_atoms_all[index]
        adj_original = self.adj_matrix[index]
        adj = np.zeros_like(adj_original)
        for v, i in self.edge_relabel_map.items():
            adj[adj_original == v] = i

        node_feature_matrix = self.node_feature_matrix[index]
        labels = np.array(
            [self.node_relabel_map[v] if v != 0 else 0
             for v in node_feature_matrix.flatten()])

        # here zeros are padded for small graph
        x = np.zeros((self.max_num_nodes, self.max_prev_nodes), dtype="float32")
        # TODO: is 1 a correct number?
        x[0, :] = 1  # the first input token is all ones
        # here zeros are padded for small graph
        y = np.zeros((self.max_num_nodes, self.max_prev_nodes), dtype="float32")
        c_in = np.zeros(self.max_num_nodes, dtype="int")
        c_out = np.zeros(self.max_num_nodes, dtype="int")
        if self.random_order:
            order = np.random.permutation(num_nodes)
            adj = adj[np.ix_(order, order)]
            labels = labels[order]

        adj_matrix = np.asmatrix(adj)
        G = nx.from_numpy_matrix(adj_matrix)

        if self.start_node_label is None:
            start_idx = np.random.randint(num_nodes)
        else:
            start_idx = np.random.choice(
                np.where(labels == self.start_node_label)[0])

        # BFS ordering
        order = np.array(bfs_seq(G, start_idx))
        adj = adj[np.ix_(order, order)]
        labels = labels[order]

        # TODO: is copy needed here?
        adj_encoded = encode_adj(adj.copy(), max_prev_node=self.max_prev_nodes)
        # get x and y and adj
        # for small graph the rest are zero padded
        y[0:adj_encoded.shape[0], :] = adj_encoded
        x[1:adj_encoded.shape[0] + 1, :] = adj_encoded
        # Important: first node must be fixed to specific label
        #  when training with vertex labels
        c_in[0:adj_encoded.shape[0] + 1] = labels
        c_out[0:adj_encoded.shape[0]] = labels[1:]
        return {'x': x, 'y': y, 'num_nodes': num_nodes,
                'c_in': c_in, 'c_out': c_out}

# TODO: variable length batching

import os, glob

import torch
import numpy as np
import networkx as nx
import pickle

from openchem.utils.graph import Graph

from torch.utils.data import Dataset
from openchem.data.utils import read_smiles_property_file, sanitize_smiles
from openchem.utils.utils_3d import calculate_xyz, calculate_zmat
from rdkit import Chem

from .graph_utils import bfs_seq, encode_adj

import torchani
from torchani.nn import SpeciesConverter
from torchani import AEVComputer


class GraphDataset(Dataset):
    def __init__(self,
                 get_atomic_attributes,
                 node_attributes,
                 filename,
                 cols_to_read,
                 delimiter=',',
                 get_bond_attributes=None,
                 edge_attributes=None,
                 restrict_min_atoms=-1,
                 restrict_max_atoms=-1,
                 kekulize=True,
                 file_format="smi",
                 addHs=False,
                 has_3D=False,
                 allowed_atoms=None,
                 return_smiles=False,
                 **kwargs):
        super(GraphDataset, self).__init__()
        assert (get_bond_attributes is None) == (edge_attributes is None)
        self.return_smiles = return_smiles
        self.restrict_min_atoms = restrict_min_atoms
        self.restrict_max_atoms = restrict_max_atoms
        self.kekulize = kekulize
        self.addHs = addHs
        self.has_3D = has_3D

        if file_format == "pickled":
            data = pickle.load(open(kwargs["pickled"], "rb"))

            # this cleanup must be consistent with sanitize_smiles
            mn, mx = restrict_min_atoms, restrict_max_atoms
            indices = [i for i, n in enumerate(data["num_atoms_all"]) if (n >= mn or mn < 0) and (n <= mx or mx < 0)]
            data = {
                key: value[indices] if isinstance(value, np.ndarray) else [value[i] for i in indices]
                for key, value in data.items()
            }

            self.num_atoms_all = data["num_atoms_all"]
            self.target = data["target"]
            self.smiles = data["smiles"]
        elif file_format == "smi":
            data_set = read_smiles_property_file(filename, cols_to_read, delimiter)
            data = data_set[0]
            if len(cols_to_read) == 1:
                target = None
            else:
                target = data_set[1:]
            clean_smiles, clean_idx, num_atoms, max_len = sanitize_smiles(data,
                                                                          min_atoms=restrict_min_atoms,
                                                                          max_atoms=restrict_max_atoms,
                                                                          return_num_atoms=True,
                                                                          return_max_len=True)
            self.max_len = max_len
            if target is not None:
                target = np.asarray(target, dtype=np.float).T
            clean_smiles = [clean_smiles[i] for i in clean_idx]
            num_atoms = [num_atoms[i] for i in clean_idx]
            self.clean_idx = clean_idx
            if target is not None:
                self.target = target[clean_idx, :]
            else:
                self.target = None
            self.smiles = clean_smiles
            self.num_atoms_all = num_atoms
        elif file_format == "sdf":
            filenames = []
            os.chdir("/home/Work/data/enamine_hll-500/")
            for file in glob.glob("*.sdf"):
                filenames.append(file)
            self.num_atoms_all = []
            smiles = []
            rd_mols = []
            for f in [filenames[10]]:
                print(f)
                supplier = Chem.SDMolSupplier(f, False, False)
                n = len(supplier)
                for i in range(n):
                    mol = supplier[i]
                    anum = [(a.GetAtomicNum() in allowed_atoms.keys()) for a in mol.GetAtoms()]
                    if sum(anum) == len(anum):
                        n = mol.GetNumAtoms()
                        x_coord = []
                        y_coord = []
                        z_coord = []
                        for k in range(n):
                            pos = mol.GetConformer().GetAtomPosition(k)
                            x_coord.append(pos.x) 
                            y_coord.append(pos.y)
                            z_coord.append(pos.z)
                        if np.linalg.norm(z_coord, ord=2) > 1.0:
                            rd_mols.append(mol)
                            smiles.append(Chem.MolToSmiles(mol))
                            self.num_atoms_all.append(n)
            self.smiles = smiles
            self.rd_mols = rd_mols
            self.target = np.ones(len(self.smiles))
        else:
            raise NotImplementedError()

        self.max_size = max(self.num_atoms_all)
        self.node_attributes = node_attributes
        self.edge_attributes = edge_attributes
        self.get_atomic_attributes = get_atomic_attributes
        self.get_bond_attributes = get_bond_attributes

    def __len__(self):
        if self.has_3D:
            return len(self.rd_mols)
        else:
            return len(self.smiles)

    def __getitem__(self, index):
        
        if self.has_3D:
            rdmol = self.rd_mols[index]
            graph = Graph(rdmol, self.max_size, self.get_atomic_attributes, self.get_bond_attributes, kekulize=self.kekulize, 
                          has_3D=self.has_3D, addHs=self.addHs, from_rdmol=True)
        else:
            sm = self.smiles[index]
            if self.return_smiles:
                object = sm + " " * (self.max_len - len(sm) + 1)
                object = [ord(c) for c in object]
            graph = Graph(sm, self.max_size, self.get_atomic_attributes, self.get_bond_attributes, kekulize=self.kekulize)
        node_feature_matrix = graph.get_node_feature_matrix(self.node_attributes, self.max_size)

        # TODO: remove diagonal elements from adjacency matrix
        if self.get_bond_attributes is None:
            adj_matrix = graph.adj_matrix
        else:
            adj_matrix = graph.get_edge_attr_adj_matrix(self.edge_attributes, self.max_size)
        if self.has_3D:
            if self.target is not None:
                sample = {
                    'adj_matrix': adj_matrix.astype('float32'),
                    'node_feature_matrix': node_feature_matrix.astype('float32'),
                    'labels': self.target[index].astype('float32'),
                    'xyz': graph.xyz#graph.xyz#(graph.xyz - mean_coord) / std_coord
                }
            elif self.target is None and not self.return_smiles:
                sample = {
                    'adj_matrix': adj_matrix.astype('float32'),
                    'node_feature_matrix': node_feature_matrix.astype('float32'),
                    'xyz': graph.xyz  # graph.xyz#(graph.xyz - mean_coord) / std_coord
                }
            elif self.return_smiles:
                sample = {
                    'adj_matrix': adj_matrix.astype('float32'),
                    'node_feature_matrix': node_feature_matrix.astype('float32'),
                    'xyz': graph.xyz,
                    'object': object
                }
        else:
            if self.target is not None:
                sample = {
                    'adj_matrix': adj_matrix.astype('float32'),
                    'node_feature_matrix': node_feature_matrix.astype('float32'),
                    'labels': self.target[index].astype('float32')
                }
            elif self.target is None and not self.return_smiles:
                sample = {
                    'adj_matrix': adj_matrix.astype('float32'),
                    'node_feature_matrix': node_feature_matrix.astype('float32'),
                }
            elif self.return_smiles:
                sample = {
                    'adj_matrix': adj_matrix.astype('float32'),
                    'node_feature_matrix': node_feature_matrix.astype('float32'),
                    'object': np.array(object)
                }
        return sample


class BFSGraphDataset(GraphDataset):
    def __init__(self, *args, **kwargs):
        super(BFSGraphDataset, self).__init__(*args, **kwargs)
        self.random_order = kwargs["random_order"]
        self.max_prev_nodes = kwargs["max_prev_nodes"]
        self.num_edge_classes = kwargs
        self.max_num_nodes = max(self.num_atoms_all)
        assert self.max_num_nodes == self.restrict_max_atoms or \
            self.restrict_max_atoms < 0, \
            "restrict_max_atoms number is too high: " + \
            "maximum number of nodes in molecules is {:d}".format(
                self.max_num_nodes
            )

        original_start_node_label = kwargs.get("original_start_node_label", None)

        if "node_relabel_map" not in kwargs:
            # define relabelling from Periodic Table numbers to {0, 1, ...}
            unique_labels = set()
            for index in range(len(self)):
                sample = super(BFSGraphDataset, self).__getitem__(index)
                node_feature_matrix = sample['node_feature_matrix']
                adj_matrix = sample['adj_matrix']

                labels = set(node_feature_matrix.flatten().tolist())
                unique_labels.update(labels)

            # discard 0 padding
            unique_labels.discard(0)

            self.node_relabel_map = {v: i for i, v in enumerate(sorted(unique_labels))}
        else:
            self.node_relabel_map = kwargs["node_relabel_map"]
        self.inverse_node_relabel_map = {i: v for v, i in self.node_relabel_map.items()}
       
        
        if original_start_node_label is not None:
            self.start_node_label = \
                self.node_relabel_map[original_start_node_label]
        else:
            self.start_node_label = None

        if "edge_relabel_map" not in kwargs:
            raise NotImplementedError()
        else:
            self.edge_relabel_map = kwargs["edge_relabel_map"]
        self.inverse_edge_relabel_map = {i: v for v, i in sorted(self.edge_relabel_map.items(), reverse=True)}

        self.num_node_classes = len(self.inverse_node_relabel_map)
        self.num_edge_classes = len(self.inverse_edge_relabel_map)
        if self.has_3D:
            self.const_file = kwargs["const_file"]
            consts = torchani.neurochem.Constants(self.const_file)
            self.aev_computer = AEVComputer(**consts)
            self.species_converter = SpeciesConverter(consts.species)

    def __getitem__(self, index):
        sample = super(BFSGraphDataset, self).__getitem__(index)
        adj_original = sample['adj_matrix']
        node_feature_matrix = sample['node_feature_matrix']
        num_nodes = self.num_atoms_all[index]
        if self.has_3D:
            xyz = sample['xyz']
        
        adj_original = adj_original.reshape(adj_original.shape[:2])
        adj = np.zeros_like(adj_original)
        for v, i in self.edge_relabel_map.items():
            adj[adj_original == v] = i
        
        labels = np.array([self.node_relabel_map[v] if v != 0 else 0 for v in node_feature_matrix.flatten()])
    
        node_feature_matrix = node_feature_matrix.flatten()
        
        if self.random_order:
            order = np.random.permutation(num_nodes)
            adj = adj[np.ix_(order, order)]
            labels = labels[order]
            if self.has_3D:
                xyz = xyz[order, :]

        adj_matrix = np.asmatrix(adj)
        G = nx.from_numpy_matrix(adj_matrix)

        if self.start_node_label is None:
            start_idx = np.random.randint(num_nodes)
        else:
            start_idx = np.random.choice(np.where(labels == self.start_node_label)[0])

        # BFS ordering
        order = np.array(bfs_seq(G, start_idx))
        adj = adj[np.ix_(order, order)]
        labels = labels[order]
        node_feature_matrix = node_feature_matrix[order]
        # reordering xyz matrix of coordinates if it exists
        if self.has_3D:
            xyz_bfs = xyz[order, :]
            num_atoms = order.shape[0]
            _, _, d_list, r_connect, a_connect, d_connect = calculate_zmat(xyz_bfs)
            d_array = np.round(np.array(d_list) + 180.0)
            classes = np.zeros_like(d_array)
            for i in range(36):
                classes[d_array >= 10.0*i] = i + 1
            padding_zeros = np.zeros((self.max_size - num_atoms, 3))
            classes = np.concatenate((np.zeros((2)), classes, -1*np.ones(self.max_size - num_atoms + 1)))
            xyz_bfs = np.concatenate((xyz_bfs, padding_zeros), axis=0)
        ii, jj = np.where(adj)
        max_prev_nodes_local = np.abs(ii - jj).max()
        
        # TODO: remove constant 1008 from here
        if self.has_3D:
            aevs = torch.zeros((self.max_num_nodes, self.max_num_nodes, 1008))
            for i in range(1, num_nodes+1):
                anum = torch.tensor(node_feature_matrix[:i]).unsqueeze(0).to(dtype=torch.long)
                coords = torch.from_numpy(xyz_bfs[:i, :]).unsqueeze(0).to(dtype=torch.float)[:, :anum.size()[1], :]
                _input = self.species_converter((anum, coords))
                aevs_ = self.aev_computer(_input)
                aevs[i-1, :i, :] = aevs_.aevs.squeeze(0)
        
        # TODO: is copy needed here?
        adj_encoded = encode_adj(adj.copy(), max_prev_node=self.max_prev_nodes)

        adj_encoded = torch.tensor(adj_encoded, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)

        x = torch.zeros((self.max_num_nodes, self.max_prev_nodes), dtype=torch.float)
        # TODO: the first input token is all ones?
        x[0, :] = 1.
        y = torch.zeros((self.max_num_nodes, self.max_prev_nodes), dtype=torch.long)
        c_in = torch.zeros(self.max_num_nodes, dtype=torch.long)
        c_out = -1 * torch.ones(self.max_num_nodes, dtype=torch.long)

        y[:num_nodes - 1, :] = adj_encoded.to(dtype=torch.long)
        x[1:num_nodes, :] = adj_encoded
        c_in[:num_nodes] = labels
        c_out[:num_nodes - 1] = labels[1:]
        if 'xyz' in sample.keys():
            return {
                'x': x,
                'y': y,
                'num_nodes': num_nodes,
                'c_in': c_in,
                'c_out': c_out,
                'max_prev_nodes_local': max_prev_nodes_local, 
                'd_classes': classes, 
                #'xyz_coord': xyz_bfs - xyz_bfs[0],
                'aevs': aevs
            }  
        else:       
            return {
                'x': x,
                'y': y,
                'num_nodes': num_nodes,
                'c_in': c_in,
                'c_out': c_out,
                'max_prev_nodes_local': max_prev_nodes_local
            }

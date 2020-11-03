from rdkit import Chem
import numpy as np


class Attribute:
    def __init__(self, attr_type, name, one_hot=True, values=None):
        if attr_type not in ['node', 'edge']:
            raise ValueError('Invalid value for attribute type: must be "node" ' 'or "edge"')
        self.attr_type = attr_type
        self.name = name
        if values is not None:
            self.n_values = len(values)
            self.attr_values = values
        self.one_hot = one_hot
        if self.one_hot:
            self.one_hot_dict = {}
            for i in range(self.n_values):
                tmp = np.zeros(self.n_values)
                tmp[i] = 1
                self.one_hot_dict[self.attr_values[i]] = tmp


class Node:
    def __init__(self, idx, rdmol, get_atom_attributes, has_3D=False):
        rdatom = rdmol.GetAtoms()[idx]
        self.node_idx = idx
        self.atom_type = rdatom.GetAtomicNum()
        self.attributes_dict = get_atom_attributes(rdatom)
        if has_3D:
            pos = rdmol.GetConformer().GetAtomPosition(idx)
            self.pos_x = pos.x
            self.pos_y = pos.y
            self.pos_z = pos.z

class Edge:
    def __init__(self, rdbond, get_bond_attributes=None):
        self.begin_atom_idx = rdbond.GetBeginAtomIdx()
        self.end_atom_idx = rdbond.GetEndAtomIdx()
        if get_bond_attributes is not None:
            self.attributes_dict = get_bond_attributes(rdbond)


class Graph:
    """Describes an undirected graph class"""
    def __init__(self, mol, max_size, get_atom_attributes, get_bond_attributes=None, kekulize=True, addHs=False, 
                 has_3D=False, from_rdmol=False):
        self.kekulize = kekulize
        self.addHs = addHs
        self.has_3D = has_3D
        if from_rdmol:
            self.smiles = Chem.MolToSmiles(mol)
        else:
            self.smiles = mol
        if from_rdmol:
            rdmol = mol
        else:
            rdmol = Chem.MolFromSmiles(mol)
            if self.addHs:
                rdmol = Chem.AddHs(rdmol)
        if kekulize:
            Chem.Kekulize(rdmol)
        self.num_nodes = rdmol.GetNumAtoms()
        self.num_edges = rdmol.GetNumBonds()

        self.nodes = []
        if has_3D:
            self.xyz = np.zeros((max_size, 3))
        num_atoms = rdmol.GetNumAtoms()
        for k in range(num_atoms):
            cur_node = Node(k, rdmol, get_atom_attributes, has_3D)
            self.nodes.append(cur_node)
            if has_3D:
                self.xyz[k, 0] = cur_node.pos_x
                self.xyz[k, 1] = cur_node.pos_y
                self.xyz[k, 2] = cur_node.pos_z

        adj_matrix = np.eye(self.num_nodes)

        self.edges = []
        for _, bond in enumerate(rdmol.GetBonds()):
            cur_edge = Edge(bond, get_bond_attributes)
            self.edges.append(cur_edge)
            adj_matrix[cur_edge.begin_atom_idx, cur_edge.end_atom_idx] = 1.0
            adj_matrix[cur_edge.end_atom_idx, cur_edge.begin_atom_idx] = 1.0
        self.adj_matrix = np.zeros((max_size, max_size))
        self.adj_matrix[:self.num_nodes, :self.num_nodes] = adj_matrix
        if get_bond_attributes is not None and len(self.edges) > 0:
            tmp = self.edges[0]
            self.n_attr = len(tmp.attributes_dict.keys())

    def get_node_attr_adj_matrix(self, attr):
        node_attr_adj_matrix = np.zeros((self.num_nodes, self.num_nodes, attr.n_values))
        attr_one_hot = []
        node_idx = []

        for node in self.nodes:
            tmp = attr.one_hot_dict[node.attributes_dict[attr.name]]
            attr_one_hot.append(tmp)
            node_attr_adj_matrix[node.node_idx, node.node_idx] = tmp
            node_idx.append(node.node_idx)

        for edge in self.edges:
            begin = edge.begin_atom_idx
            end = edge.end_atom_idx
            begin_one_hot = attr_one_hot[node_idx.index(begin)]
            end_one_hot = attr_one_hot[node_idx.index(end)]
            node_attr_adj_matrix[begin, end, :] = (begin_one_hot + end_one_hot) / 2

        return node_attr_adj_matrix

    def get_edge_attr_adj_matrix(self, all_atr_dict, max_size):
        fl = True
        for edge in self.edges:
            begin = edge.begin_atom_idx
            end = edge.end_atom_idx
            cur_features = []
            for attr_name in edge.attributes_dict.keys():
                cur_attr = all_atr_dict[attr_name]
                if cur_attr.one_hot:
                    cur_features += list(cur_attr.one_hot_dict[edge.attributes_dict[cur_attr.name]])
                else:
                    cur_features += [edge.attributes_dict[cur_attr.name]]
            cur_features = np.array(cur_features)
            attr_len = len(cur_features)
            if fl:
                edge_attr_adj_matrix = np.zeros((max_size, max_size, attr_len))
                fl = False
            edge_attr_adj_matrix[begin, end, :] = cur_features
            edge_attr_adj_matrix[end, begin, :] = cur_features

        return edge_attr_adj_matrix

    def get_node_feature_matrix(self, all_atr_dict, max_size):
        features = []
        for node in self.nodes:
            cur_features = []
            for attr_name in node.attributes_dict.keys():
                cur_attr = all_atr_dict[attr_name]
                if cur_attr.one_hot:
                    cur_features += list(cur_attr.one_hot_dict[node.attributes_dict[cur_attr.name]])
                else:
                    cur_features += [node.attributes_dict[cur_attr.name]]
            features.append(cur_features)

        features = np.array(features)
        padded_features = np.zeros((max_size, features.shape[1]))
        padded_features[:features.shape[0], :features.shape[1]] = features
        return padded_features
    
    
    def xyz_to_zmat(self):
        raise NotImplementedError()
        
    def zmat_to_xyz(self):
        raise NotImplementedError()
    
        


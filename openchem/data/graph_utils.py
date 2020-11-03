# Graph utils from original GraphRNN implementation
# https://github.com/JiaxuanYou/graph-generation

import networkx as nx
import numpy as np

from rdkit import Chem
from rdkit.Geometry.rdGeometry import Point3D


def bfs_seq(G, start_id, return_bfs=False):
    '''
    get a bfs node sequence
    :param G:
    :param start_id:
    :return:
    '''
    dictionary = dict(nx.bfs_successors(G, start_id))
    start = [start_id]
    output = [start_id]
    while len(start) > 0:
        next = []
        while len(start) > 0:
            current = start.pop(0)
            neighbor = dictionary.get(current)
            if neighbor is not None:
                #### a wrong example, should not permute here!
                # shuffle(neighbor)
                next = next + neighbor
        output = output + next
        start = next
    if return_bfs:
        remap = {}
        for i in range(len(output)):
            remap[output[i]] = i
        bfs_dict = {}
        for key in dictionary.keys():
            bfs_dict[remap[key]] = []
            for j in range(len(dictionary[key])):
                bfs_dict[remap[key]].append(remap[dictionary[key][j]])
        nodes = list(range(len(output)))
        parents = list(np.zeros(len(nodes), dtype="int"))
        zeros_neighb = []
        for key in bfs_dict.keys():
            for j in range(len(bfs_dict[key])):
                parents[bfs_dict[key][j]] = key
        return output, nodes, parents, bfs_dict
    else:
        return output


def encode_adj(adj, max_prev_node=10, is_full=False):
    '''

    :param adj: n*n, rows means time step, while columns are input dimension
    :param max_degree: we want to keep row number, but truncate column numbers
    :return:
    '''
    if is_full:
        max_prev_node = adj.shape[0] - 1

    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0:n - 1]

    # use max_prev_node to truncate
    # note: now adj is a (n-1)*(n-1) matrix
    adj_output = np.zeros((adj.shape[0], max_prev_node))
    for i in range(adj.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + input_start - input_end
        output_end = max_prev_node
        adj_output[i, output_start:output_end] = adj[i, input_start:input_end]
        adj_output[i, :] = adj_output[i, :][::-1]  # reverse order

    return adj_output


def decode_adj(adj_output):
    '''
        recover to adj from adj_output
        note: here adj_output have shape (n-1)*m
    '''
    max_prev_node = adj_output.shape[1]
    adj = np.zeros((adj_output.shape[0], adj_output.shape[0]))
    for i in range(adj_output.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + max(0, i - max_prev_node + 1) - (i + 1)
        output_end = max_prev_node
        adj[i, input_start:input_end] = adj_output[i, ::-1][output_start:output_end]  # reverse order
    adj_full = np.zeros((adj_output.shape[0] + 1, adj_output.shape[0] + 1))
    n = adj_full.shape[0]
    adj_full[1:n, 0:n - 1] = np.tril(adj, 0)
    adj_full = adj_full + adj_full.T

    return adj_full


def decode_adj_new(adj_output):
    '''
        recover to adj from adj_output
        note: here adj_output have shape (n-1)*m
    '''
    num_nodes = adj_output.shape[0] + 1
    adj_full = np.zeros((num_nodes, num_nodes), dtype=adj_output.dtype)

    for i, d in enumerate(adj_output.T):
        a = np.diag(d, k=i + 1)[i:, i:]
        adj_full = adj_full + a + a.T

    return adj_full


def SmilesFromGraphs(node_list, adjacency_matrix, remap=None):
    """
    Converts molecular graph to SMILES string
    :param node_list:
    :param adjacency_matrix:
    :param remap: edge inverse relabel map
    :return:
    """
    # create empty editable mol object
    mol = Chem.RWMol()

    # add atoms to mol and keep track of index
    node_to_idx = {}
    for i in range(len(node_list)):
        a = Chem.Atom(node_list[i])
        molIdx = mol.AddAtom(a)
        node_to_idx[i] = molIdx

    # add bonds between adjacent atoms
    xx, yy = np.where(np.triu(adjacency_matrix, k=1))
    vv = adjacency_matrix[xx, yy]
    for ix, iy, bond in zip(xx, yy, vv):
        # add relevant bond type (there are many more of these)
        if remap is not None:
            bond = remap[bond]
        if bond == 1.:
            bond_type = Chem.rdchem.BondType.SINGLE
            mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
        elif bond == 1.5:
            bond_type = Chem.rdchem.BondType.AROMATIC
            mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
        elif bond == 2.:
            bond_type = Chem.rdchem.BondType.DOUBLE
            mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
        elif bond == 3.:
            bond_type = Chem.rdchem.BondType.TRIPLE
            mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
        else:
            raise ValueError("Invalid bond type in matrix")

    # Convert RWMol to Mol object
    mol = mol.GetMol()
    #Chem.Kekulize(mol)

    # Convert RWMol to SMILES
    smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)

    return smiles, mol


def RDMolFromGraphs(node_list, adjacency_matrix, xyz_coord, remap=None):
    #xyz_mean = [0.5567917341433811, 0.10013086135351022, 0.006413393431574624]
    #xyz_std = [3.40762324684836, 2.6455335437418093, 2.3609727916942096]
    #print(xyz_coord)
    #xyz_coord = xyz_coord * xyz_std + xyz_mean
    #print(xyz_coord)
    # create empty editable mol object
    mol = Chem.RWMol()
    conformer = Chem.Conformer(len(node_list))
    #print(xyz_coord)
    # add atoms to mol and keep track of index
    node_to_idx = {}
    for i in range(len(node_list)):
        a = Chem.Atom(node_list[i])
        atom_position = Point3D(xyz_coord[i, 0], xyz_coord[i, 1], xyz_coord[i, 2])
        #print(a)
        conformer.SetAtomPosition(i, atom_position)
        molIdx = mol.AddAtom(a)
        node_to_idx[i] = molIdx

    # add bonds between adjacent atoms
    xx, yy = np.where(np.triu(adjacency_matrix, k=1))
    vv = adjacency_matrix[xx, yy]
    for ix, iy, bond in zip(xx, yy, vv):
        # add relevant bond type (there are many more of these)
        if remap is not None:
            bond = remap[bond]
        if bond == 1.:
            bond_type = Chem.rdchem.BondType.SINGLE
            mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
        elif bond == 1.5:
            bond_type = Chem.rdchem.BondType.AROMATIC
            mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
        elif bond == 2.:
            bond_type = Chem.rdchem.BondType.DOUBLE
            mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
        elif bond == 3.:
            bond_type = Chem.rdchem.BondType.TRIPLE
            mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
        else:
            raise ValueError("Invalid bond type in matrix")

    # Convert RWMol to Mol object
    mol = mol.GetMol()
    Chem.Kekulize(mol)
    Chem.SanitizeMol(mol)
    
    # Convert RWMol to SMILES
    # Chem.RemoveHs(mol)
    smiles = Chem.MolToSmiles(mol, kekuleSmiles=False)
    #print(smiles)
    
    # Add 3D conformation to RDKit molecule
    added_conformer = mol.AddConformer(conformer, assignId=True)

    return smiles, mol
    

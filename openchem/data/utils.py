# TODO: packed variable length sequence

import time
import math
import numpy as np
import csv
import warnings

from rdkit import Chem
from rdkit import DataStructs

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from openchem.data.smiles_enumerator import SmilesEnumerator

from rdkit import rdBase
from rdkit import Chem
from openchem.utils.graph import Graph
rdBase.DisableLog('rdApp.error')


class DummyDataset(Dataset):
    def __init__(self, size=1):
        super(DummyDataset, self).__init__()
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {
            'x': torch.zeros(1),
            'y': torch.zeros(1),
            'num_nodes': torch.zeros(1),
            'c_in': torch.zeros(1),
            'c_out': torch.zeros(1),
            'max_prev_nodes_local': torch.zeros(1)
        }


class DummyDataLoader(object):
    def __init__(self, batch_size):
        self.batch_size = 32  #batch_size
        self.current = 0

    def __iter__(self):
        return self

    def __len__(self):
        return self.batch_size

    def __next__(self):
        if self.current < self.batch_size:
            self.current += 1
            return None
        else:
            self.current = 0
            raise StopIteration


def cut_padding(samples, lengths, padding='left'):
    max_len = lengths.max(dim=0)[0].cpu().numpy()
    if padding == 'right':
        cut_samples = samples[:, :max_len]
    elif padding == 'left':
        total_len = samples.size()[1]
        cut_samples = samples[:, total_len - max_len:]
    else:
        raise ValueError('Invalid value for padding argument. Must be right' 'or left')
    return cut_samples


def seq2tensor(seqs, tokens, flip=True):
    tensor = np.zeros((len(seqs), len(seqs[0])))
    for i in range(len(seqs)):
        for j in range(len(seqs[i])):
            if seqs[i][j] in tokens:
                tensor[i, j] = tokens.index(seqs[i][j])
            else:
                tokens = tokens + seqs[i][j]
                tensor[i, j] = tokens.index(seqs[i][j])
    if flip:
        tensor = np.flip(tensor, axis=1).copy()
    return tensor, tokens


def pad_sequences(seqs, max_length=None, pad_symbol=' '):
    if max_length is None:
        max_length = -1
        for seq in seqs:
            max_length = max(max_length, len(seq))
    lengths = []
    for i in range(len(seqs)):
        cur_len = len(seqs[i])
        lengths.append(cur_len)
        seqs[i] = seqs[i] + pad_symbol * (max_length - cur_len)
    return seqs, lengths


def create_loader(dataset, batch_size, shuffle=True, num_workers=1, pin_memory=False, sampler=None):
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             sampler=sampler)

    return data_loader


def sanitize_smiles(smiles,
                    canonize=True,
                    min_atoms=-1,
                    max_atoms=-1,
                    return_num_atoms=False,
                    allowed_tokens=None,
                    allow_charges=False,
                    return_max_len=False,
                    logging="warn"):
    """
    Takes list of SMILES strings and returns list of their sanitized versions.
    For definition of sanitized SMILES check
    http://www.rdkit.org/docs/api/rdkit.Chem.rdmolops-module.html#SanitizeMol
        Args:
            smiles (list): list of SMILES strings
            canonize (bool): parameter specifying whether to return
            canonical SMILES or not.
            min_atoms (int): minimum allowed number of atoms
            max_atoms (int): maxumum allowed number of atoms
            return_num_atoms (bool): return additional array of atom numbers
            allowed_tokens (iterable, optional): allowed tokens set
            allow_charges (bool): allow nonzero charges of atoms
            logging ("warn", "info", "none"): logging level
        Output:
            new_smiles (list): list of SMILES and NaNs if SMILES string is
            invalid or unsanitized.
            If 'canonize = True', return list of canonical SMILES.
        When 'canonize = True' the function is analogous to:
        canonize_smiles(smiles, sanitize=True).
    """
    assert logging in ["warn", "info", "none"]

    new_smiles = []
    idx = []
    num_atoms = []
    smiles_lens = []
    for i in range(len(smiles)):
        sm = smiles[i]
        mol = Chem.MolFromSmiles(sm, sanitize=False)
        sm_new = Chem.MolToSmiles(mol) if canonize and mol is not None else sm
        good = mol is not None
        if good and allowed_tokens is not None:
            good &= all([t in allowed_tokens for t in sm_new])

        if good and not allow_charges:
            good &= all([a.GetFormalCharge() == 0 for a in mol.GetAtoms()])

        if good:
            n = mol.GetNumAtoms()
            if (n < min_atoms and min_atoms > -1) or (n > max_atoms > -1):
                good = False
        else:
            n = 0

        if good:
            new_smiles.append(sm_new)
            idx.append(i)
            num_atoms.append(n)
            smiles_lens.append(len(sm_new))
        else:
            new_smiles.append(' ')
            num_atoms.append(0)

    smiles_set = set(new_smiles)
    num_unique = len(smiles_set) - ('' in smiles_set)
    if len(idx) > 0:
        valid_unique_rate = float(num_unique) / len(idx)
        invalid_rate = 1.0 - float(len(idx)) / len(smiles)
    else:
        valid_unique_rate = 0.0
        invalid_rate = 1.0
    num_bad = len(smiles) - len(idx)

    if len(idx) != len(smiles) and logging == "warn":
        warnings.warn('{:d}/{:d} unsanitized smiles ({:.1f}%)'.format(num_bad, len(smiles), 100 * invalid_rate))
    elif logging == "info":
        print("Valid: {}/{} ({:.2f}%)".format(len(idx), len(smiles), 100 * (1 - invalid_rate)))
        print("Unique valid: {:.2f}%".format(100 * valid_unique_rate))

    if return_num_atoms and return_max_len:
        return new_smiles, idx, num_atoms, max(smiles_lens)
    elif return_num_atoms and not return_max_len:
        return new_smiles, idx, num_atoms
    elif not return_num_atoms and return_max_len:
        return new_smiles, idx, max(smiles_lens)
    else:
        return new_smiles, idx

def canonize_smiles(smiles, sanitize=True):
    """
    Takes list of SMILES strings and returns list of their canonical SMILES.
        Args:
            smiles (list): list of SMILES strings
            sanitize (bool): parameter specifying whether to sanitize
            SMILES or not.
            For definition of sanitized SMILES check
            www.rdkit.org/docs/api/rdkit.Chem.rdmolops-module.html#SanitizeMol
        Output:
            new_smiles (list): list of canonical SMILES and NaNs
            if SMILES string is invalid or unsanitized
            (when 'sanitize = True')
        When 'sanitize = True' the function is analogous to:
        sanitize_smiles(smiles, canonize=True).
    """
    new_smiles = []
    idx = []
    for i in range(len(smiles)):
        sm = smiles[i]
        try:
            new_smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(sm, sanitize=sanitize)))
            idx.append(i)
        except:
            new_smiles.append('')

        if len(idx) != len(smiles):
            invalid_rate = 1.0 - len(idx) / len(smiles)
            warnings.warn('Proportion of unsanitized smiles is %.3f ' % (invalid_rate))
    return new_smiles


def save_smi_to_file(filename, smiles, unique=True):
    """
    Takes path to file and list of SMILES strings and writes SMILES
    to the specified file.
        Args:
            filename (str): path to the file
            smiles (list): list of SMILES strings
            unique (bool): parameter specifying whether to write
            only unique copies or not.
        Output:
            success (bool): defines whether operation
            was successfully completed or not.
       """
    if unique:
        smiles = list(set(smiles))
    else:
        smiles = list(smiles)
    f = open(filename, 'w')
    for mol in smiles:
        f.writelines([mol, '\n'])
    f.close()
    return f.closed


def read_smi_file(filename, unique=True):
    """
    Reads SMILES from file. File must contain one SMILES string per line
    with \n token in the end of the line.
    Args:
        filename (str): path to the file
        unique (bool): return only unique SMILES
    Returns:
        smiles (list): list of SMILES strings from specified file.
        success (bool): defines whether operation
        was successfully completed or not.
    If 'unique=True' this list contains only unique copies.
    """
    f = open(filename, 'r')
    molecules = []
    for line in f:
        molecules.append(line[:-1])
    if unique:
        molecules = list(set(molecules))
    else:
        molecules = list(molecules)
    f.close()
    return molecules, f.closed


def get_tokens(smiles, tokens=None):
    """
    Returns list of unique tokens, token-2-index dictionary and
    number of unique tokens from the list of SMILES
    Args:
        smiles (list): list of SMILES strings to tokenize.
        tokens (string): string of tokens or None.
        If none will be extracted from dataset.
    Returns:
        tokens (list): list of unique tokens/SMILES alphabet.
        token2idx (dict): dictionary mapping token to its index.
        num_tokens (int): number of unique tokens.
    """
    if tokens is None:
        tokens = list(set(''.join(smiles)))
        tokens = sorted(tokens)
        tokens = ''.join(tokens)
    token2idx = dict((token, i) for i, token in enumerate(tokens))
    num_tokens = len(tokens)
    return tokens, token2idx, num_tokens


def augment_smiles(smiles, labels, n_augment=5):
    smiles_augmentation = SmilesEnumerator()
    augmented_smiles = []
    augmented_labels = []
    for i in range(len(smiles)):
        sm = smiles[i]
        for _ in range(n_augment):
            augmented_smiles.append(smiles_augmentation.randomize_smiles(sm))
            augmented_labels.append(labels[i])
        augmented_smiles.append(sm)
        augmented_labels.append(labels[i])
    return augmented_smiles, augmented_labels


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def read_smiles_property_file(path, cols_to_read, delimiter=',', keep_header=False):
    reader = csv.reader(open(path, 'r'), delimiter=delimiter)
    data = list(reader)
    if keep_header:
        start_position = 0
    else:
        start_position = 1
    assert len(data) > start_position
    data = map(list, zip(*data))
    data = [d for c, d in enumerate(data)]
    data_ = [data[c][start_position:] for c in cols_to_read]

    return data_


def save_smiles_property_file(path, smiles, labels, delimiter=','):
    f = open(path, 'w')
    n_targets = labels.shape[1]
    for i in range(len(smiles)):
        f.writelines(smiles[i])
        for j in range(n_targets):
            f.writelines(delimiter + str(labels[i, j]))
        f.writelines('\n')
    f.close()


def process_smiles(smiles,
                   sanitized=False,
                   target=None,
                   augment=False,
                   pad=True,
                   tokenize=True,
                   tokens=None,
                   flip=False,
                   allowed_tokens=None):
    if not sanitized:
        clean_smiles, clean_idx = sanitize_smiles(smiles, allowed_tokens=allowed_tokens)
        clean_smiles = [clean_smiles[i] for i in clean_idx]
        if target is not None:
            target = target[clean_idx]
    else:
        clean_smiles = smiles

    length = None
    if augment and target is not None:
        clean_smiles, target = augment_smiles(clean_smiles, target)
    if pad:
        clean_smiles, length = pad_sequences(clean_smiles)
    tokens, token2idx, num_tokens = get_tokens(clean_smiles, tokens)
    if tokenize:
        clean_smiles, tokens = seq2tensor(clean_smiles, tokens, flip)

    return clean_smiles, target, length, tokens, token2idx, num_tokens


def process_graphs(smiles,
                   node_attributes,
                   get_atomic_attributes,
                   edge_attributes,
                   get_bond_attributes=None,
                   kekulize=True):

    clean_smiles, clean_idx, num_atoms = sanitize_smiles(smiles, return_num_atoms=True)
    clean_smiles = [clean_smiles[i] for i in clean_idx]

    max_size = np.max(num_atoms)

    adj = []
    node_feat = []
    for sm in clean_smiles:
        graph = Graph(sm, max_size, get_atomic_attributes, get_bond_attributes, kekulize=kekulize)
        node_feature_matrix = graph.get_node_feature_matrix(node_attributes, max_size)
        # TODO: remove diagonal elements from adjacency matrix
        if get_bond_attributes is None:
            adj_matrix = graph.adj_matrix
        else:
            adj_matrix = graph.get_edge_attr_adj_matrix(edge_attributes, max_size)
        adj.append(adj_matrix.astype('float32'))
        node_feat.append(node_feature_matrix.astype('float32'))

    return adj, node_feat

def get_fp(smiles, n_bits=2048):
    fp = []
    processed_indices = []
    invalid_indices = []
    for i in range(len(smiles)):
        mol = smiles[i]
        tmp = np.array(mol2image(mol, n=n_bits))
        if np.isnan(tmp[0]):
            invalid_indices.append(i)
        else:
            fp.append(tmp)
            processed_indices.append(i)
    return np.array(fp, dtype="float32"), processed_indices, invalid_indices


def mol2image(x, n=2048):
    try:
        m = Chem.MolFromSmiles(x)
        fp = Chem.RDKFingerprint(m, maxPath=4, fpSize=n)
        res = np.zeros(len(fp))
        DataStructs.ConvertToNumpyArray(fp, res)
        return res
    except Exception as e:
        print(e)
        return [np.nan]

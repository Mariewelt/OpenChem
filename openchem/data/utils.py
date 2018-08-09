# TODO: packed variable length sequence

import time
import math
import numpy as np
import csv
import warnings

from rdkit import Chem

from torch.utils.data import DataLoader
from openchem.data.smiles_enumerator import SmilesEnumerator


def cut_padding(samples, lengths, padding='left'):
    max_len = lengths.max(dim=0)[0].cpu().numpy()
    if padding == 'right':
        cut_samples = samples[:, :max_len]
    elif padding == 'left':
        total_len = samples.size()[1]
        cut_samples = samples[:, total_len-max_len:]
    else:
        raise ValueError('Invalid value for padding argument. Must be right'
                         'or left')
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
        seqs[i] = seqs[i] + pad_symbol*(max_length - cur_len)
    return seqs, lengths


def create_loader(dataset, batch_size, shuffle=True, num_workers=1,
                  pin_memory=False, sampler=None):
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                             shuffle=shuffle, num_workers=num_workers,
                             pin_memory=pin_memory, sampler=sampler)
    return data_loader


def sanitize_smiles(smiles, canonize=True):
    """
    Takes list of SMILES strings and returns list of their sanitized versions.
    For definition of sanitized SMILES check
    http://www.rdkit.org/docs/api/rdkit.Chem.rdmolops-module.html#SanitizeMol
        Args:
            smiles (list): list of SMILES strings
            canonize (bool): parameter specifying whether to return
            canonical SMILES or not.
        Output:
            new_smiles (list): list of SMILES and NaNs if SMILES string is
            invalid or unsanitized.
            If 'canonize = True', return list of canonical SMILES.
        When 'canonize = True' the function is analogous to:
        canonize_smiles(smiles, sanitize=True).
    """
    new_smiles = []
    idx = []
    for i in range(len(smiles)):
        sm = smiles[i]
        try:
            if canonize:
                new_smiles.append(
                    Chem.MolToSmiles(Chem.MolFromSmiles(sm, sanitize=False))
                )
                idx.append(i)
            else:
                new_smiles.append(sm)
                idx.append(i)
        except: 
            warnings.warn('Unsanitized SMILES string: ' + sm)
            new_smiles.append('')
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
    for sm in smiles:
        try:
            new_smiles.append(
                Chem.MolToSmiles(Chem.MolFromSmiles(sm, sanitize=sanitize))
            )
        except: 
            warnings.warn(sm + ' can not be canonized: i'
                                'nvalid SMILES string!')
            new_smiles.append('')
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
        tokens = np.sort(tokens)[::-1]
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


def read_smiles_property_file(path, cols_to_read, delimiter=',',
                              keep_header=False):
    reader = csv.reader(open(path, 'r'), delimiter=delimiter)
    data_full = np.array(list(reader))
    if keep_header:
        start_position = 0
    else:
        start_position = 1
    assert len(data_full) > start_position
    data = [[] for _ in range(len(cols_to_read))]
    for i in range(len(cols_to_read)):
        col = cols_to_read[i]
        data[i] = data_full[start_position:, col]

    return data


def save_smiles_property_file(path, smiles, labels, delimiter=','):
    f = open(path, 'w')
    n_targets = labels.shape[1]
    for i in range(len(smiles)):
        f.writelines(smiles[i])
        for j in range(n_targets):
            f.writelines(delimiter + str(labels[i, j]))
        f.writelines('\n')
    f.close()

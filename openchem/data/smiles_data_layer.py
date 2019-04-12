# TODO: packed variable length sequence

import numpy as np

from torch.utils.data import Dataset

from openchem.data.utils import read_smiles_property_file
from openchem.data.utils import sanitize_smiles, pad_sequences, seq2tensor, canonize_smiles
from openchem.data.utils import get_tokens, augment_smiles


class SmilesDataset(Dataset):
    """
    Creates dataset for SMILES-property data.
    Args:
        filename (str): string with full path to dataset file. Dataset file
            must be csv file.
        cols_to_read (list): list specifying columns to read from dataset file.
            Could be of various length, `cols_to_read[0]` will be used as index
            as index for column with SMILES, `cols_to_read[1:]` will be used as
            indices for labels values.
        delimiter (str): columns delimiter in `filename`. `default` is `,`.
        tokens (list): list of unique tokens from SMILES. If not specified, will
            be extracted from provided dataset.
        pad (bool): argument specifying whether to pad SMILES. If `true` SMILES
            will be padded from right and the flipped. `default` is `True`.
        augment (bool): argument specifying whether to augment SMILES.

    """
    def __init__(self, filename, cols_to_read, delimiter=',', tokens=None,
                 pad=True, tokenize=True, augment=False, flip=True, sanitize=True):
        super(SmilesDataset, self).__init__()
        self.tokenize = tokenize
        data = read_smiles_property_file(filename, cols_to_read, delimiter)
        smiles = data[0]
        if sanitize:
            clean_smiles, clean_idx = sanitize_smiles(smiles)
        else:
            clean_smiles = smiles
            clean_idx = list(range(len(smiles)))
        if len(data) > 1:
            target = np.array(data[1:], dtype='float')
            target = np.array(target)
            target = target.T
            self.target = target[clean_idx]
        else:
            self.target = None
        if augment:
            clean_smiles, self.target = augment_smiles(clean_smiles,
                                                       self.target)
        if pad:
            clean_smiles, self.length = pad_sequences(clean_smiles)
        tokens, self.token2idx, self.num_tokens = get_tokens(clean_smiles,
                                                                tokens)
        if tokenize:
            clean_smiles, self.tokens = seq2tensor(clean_smiles, tokens, flip)
        self.data = clean_smiles

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = {}
        sample['tokenized_smiles'] = self.data[index] 
        sample['length'] = self.length[index]
        if self.target is not None:
            sample['labels'] = self.target[index]
        return sample

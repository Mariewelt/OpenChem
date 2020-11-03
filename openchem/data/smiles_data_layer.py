# TODO: packed variable length sequence

import numpy as np

from torch.utils.data import Dataset

from openchem.data.utils import process_smiles
from openchem.data.utils import read_smiles_property_file
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
    def __init__(self,
                 filename,
                 cols_to_read,
                 delimiter=',',
                 tokens=None,
                 pad=True,
                 tokenize=True,
                 augment=False,
                 sanitize=True,
                 flip=False,
                 return_smiles=False):
        super(SmilesDataset, self).__init__()
        self.tokenize = tokenize
        self.return_smiles = return_smiles
        data = read_smiles_property_file(filename, cols_to_read, delimiter)
        if len(cols_to_read) > 1:
            assert len(cols_to_read) == len(data)
            smiles = data[0]
            target = np.array(data[1:], dtype='float')
            target = target.T
            num_targets = len(cols_to_read) - 1
            target = target.reshape((-1, num_targets))
        else:
            smiles = data[0]
            target = None
        if sanitize:
            sanitized = False
        else:
            sanitized = True
        self.data, self.target, self.length, \
            self.tokens, self.token2idx, self.num_tokens = process_smiles(
            smiles, sanitized=sanitized, target=target, augment=augment, pad=pad,
            tokenize=tokenize, tokens=tokens, flip=flip)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = {}
        if self.return_smiles:
            sample['object'] = np.array([ord(self.tokens[int(i)]) for i in self.data[index]])
        sample['tokenized_smiles'] = self.data[index]
        sample['length'] = self.length[index]
        if self.target is not None:
            sample['labels'] = self.target[index]
        return sample

import numpy as np
from openchem.data.utils import sanitize_smiles, read_smiles_property_file


class VanillaDataset(object):
    def __init__(self, filename, cols_to_read, features, delimiter=',', tokens=None):
        super(VanillaDataset, self).__init__()
        data = read_smiles_property_file(filename, cols_to_read, delimiter)
        smiles = data[0]
        target = np.array(data[1], dtype='float')
        clean_smiles, clean_idx = sanitize_smiles(smiles)
        target = np.array(target)
        self.target = target[clean_idx]

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        sample = {'tokenized_smiles': self.data[index], 'labels': self.target[index]}
        return sample

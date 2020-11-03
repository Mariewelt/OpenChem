
import numpy as np

from torch.utils.data import Dataset

from openchem.data.utils import process_smiles
from openchem.data.utils import read_smiles_property_file
from openchem.data.utils import get_tokens, augment_smiles


class FeatureDataset(Dataset):
    """
    Creates dataset for feature-property data.
    Args:
        filename (str): string with full path to dataset file. Dataset file
            must be csv file.
        cols_to_read (list): list specifying columns to read from dataset file.
            Could be of various length, `cols_to_read[0]` will be used as index
            as index for column with SMILES, `cols_to_read[1:]` will be used as
            indices for labels values.
        delimiter (str): columns delimiter in `filename`. `default` is `,`.
        get_features (python function): python function to extract features from input data
        get_feature_args (dict): additional parameters for get_features function
    """
    def __init__(self,
                 filename,
                 cols_to_read,
                 get_features,
                 delimiter=',',
                 return_smiles=False,
                 get_features_args=None):
        super(FeatureDataset, self).__init__()
        self.return_smiles = return_smiles
        self.get_features = get_features
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
        self.target = target
        features, valid_idx, invalid_idx = get_features(smiles, **get_features_args)
        self.objects = [smiles[i] for i in valid_idx]
        self.data = features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = {}
        if self.return_smiles:
            object = self.objects[index]
            sample['object'] = np.array([ord(c) for c in object])
        sample['features'] = self.data[index]
        if self.target is not None:
            sample['labels'] = self.target[index]
        return sample

import numpy as np
import pickle

from torch.utils.data import Dataset

from openchem.data.utils import read_smiles_property_file
from openchem.data.utils import sanitize_smiles, pad_sequences, seq2tensor
from openchem.data.utils import get_tokens
from openchem.data.smiles_data_layer import SmilesDataset
from openchem.data.graph_data_layer import GraphDataset


class SiameseDataset(Dataset):
    def __init__(self, filename, head1_type, head2_type, cols_to_read,
                 head1_arguments, head2_arguments):
        super(SiameseDataset, self).__init__()
        assert len(cols_to_read) == 3
        if head1_type == "smiles":
            cols_to_read = [cols_to_read[0]] + [cols_to_read[2]]
            head1_dataset = SmilesDataset(filename,
                                          cols_to_read=[0, 2],
                                          **head1_arguments)
        elif head1_type == "graphs":
            raise NotImplementedError()
            head1_dataset = GraphDataset(filename=filename,
                                         cols_to_read=[0, 2],
                                         **head1_arguments)
        else:
            raise ArgumentError
        if head2_type == "smiles":
            head2_dataset = SmilesDataset(filename,
                                          cols_to_read=[1, 2],
                                          **head2_arguments)
        elif head2_type == "graphs":
            raise NotImplementedError()
            head2_dataset = GraphDataset(filename=filename,
                                         cols_to_read=[1, 2],
                                         **head2_arguments)
        else:
            raise ArgumentError
        self.head1_dataset = head1_dataset
        self.head2_dataset = head2_dataset
        #assert len(head1_dataset) == len(head2_dataset)
        self.target = head1_dataset.target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        head1_sample = self.head1_dataset[index]
        head2_sample = self.head2_dataset[index]
        sample = {'head1': head1_sample,
                  'head2': head2_sample,
                  'labels': self.target[index]}
        return sample


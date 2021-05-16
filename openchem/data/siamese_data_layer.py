from torch.utils.data import Dataset

from openchem.data.smiles_data_layer import SmilesDataset
from openchem.data.graph_data_layer import GraphDataset


class SiameseDataset(Dataset):
    def __init__(self, filename, head1_type, head2_type, cols_to_read,
                 head1_arguments, head2_arguments):
        super(SiameseDataset, self).__init__()
        assert len(cols_to_read) == 3
        if head1_type == "protein_seq":
            head1_arguments["sanitize"] = False
        if head2_type == "protein_seq":
            head2_arguments["sanitize"] = False
        if head1_type == "mol_smiles" or head1_type == "protein_seq":
            head1_dataset = SmilesDataset(filename,
                                          cols_to_read=[cols_to_read[0], cols_to_read[2]],
                                          **head1_arguments)
        elif head1_type == "mol_graphs":
            raise NotImplementedError()
            head1_dataset = GraphDataset(filename=filename,
                                         cols_to_read=[cols_to_read[0], cols_to_read[2]],
                                         **head1_arguments)
        else:
            raise ValueError()
        if head2_type == "mol_smiles" or head2_type == "protein_seq":
            head2_dataset = SmilesDataset(filename,
                                          cols_to_read=[cols_to_read[1], cols_to_read[2]],
                                          **head2_arguments)
        elif head2_type == "mol_graphs":
            raise NotImplementedError()
            head2_dataset = GraphDataset(filename=filename,
                                         cols_to_read=[cols_to_read[1], cols_to_read[2]],
                                         **head2_arguments)
        else:
            raise ArgumentError
        self.head1_dataset = head1_dataset
        self.head2_dataset = head2_dataset
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
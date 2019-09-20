from openchem.models.Smiles2Label import Smiles2Label
from openchem.modules.embeddings.basic_embedding import Embedding
from openchem.modules.encoders.rnn_encoder import RNNEncoder
from openchem.modules.mlp.openchem_mlp import OpenChemMLP#Simple
from openchem.utils.utils import identity

from sklearn.metrics import mean_squared_error, r2_score

import torch.nn as nn

import numpy as np

from torch.optim import RMSprop, Adam
from torch.optim.lr_scheduler import ExponentialLR, StepLR, MultiStepLR


from openchem.data.utils import read_smiles_property_file
data = read_smiles_property_file('/data/masha/melting/melting_data.txt',
                                 cols_to_read=[0, 1], delimiter='\t',
                                 keep_header=False)
smiles = data[0][1:]
labels = np.array(data[1][1:], dtype='float').reshape(-1)
#print(smiles)

from openchem.data.utils import get_tokens
tokens, _, _ = get_tokens(smiles)
tokens = tokens + ' '

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(smiles, labels,
                                                    test_size=0.2,
                                                    random_state=42)

train_mean = np.mean(y_train)
train_std = np.std(y_train)
print(train_mean)
print(train_std)
print(np.min(y_train))
print(np.max(y_train))
y_train = (y_train - train_mean) / train_std
y_test = (y_test - train_mean) / train_std

from openchem.data.utils import save_smiles_property_file
save_smiles_property_file('./benchmark_datasets/melt_temp/train.smi',
                          X_train,
                          y_train.reshape(-1, 1))

save_smiles_property_file('./benchmark_datasets/melt_temp/test.smi',
                          X_test,
                          y_test.reshape(-1, 1))

from openchem.data.smiles_data_layer import SmilesDataset
train_dataset = SmilesDataset('./benchmark_datasets/melt_temp/train.smi',
                              delimiter=',', cols_to_read=[0, 1],
                              tokens=tokens, augment=True)

#train_dataset.target = train_dataset.target.reshape(-1, 1)

test_dataset = SmilesDataset('./benchmark_datasets/melt_temp/test.smi',
                            delimiter=',', cols_to_read=[0, 1],
                            tokens=tokens)
#test_dataset.target = test_dataset.target.reshape(-1, 1)

model = Smiles2Label

model_params = {
    'use_cuda': True,
    'task': 'regression',
    'random_seed': 5,
    'use_clip_grad': True,
    'max_grad_norm': 10.0,
    'batch_size': 256,
    'num_epochs': 31,
    'logdir': './logs/melt_temp',
    'print_every': 1,
    'save_every': 1,
    'train_data_layer': train_dataset,
    'val_data_layer': test_dataset,
    'eval_metrics': r2_score,
    'criterion': nn.MSELoss().cuda(),
    'optimizer': Adam,
    'optimizer_params': {
        'lr': 0.0008,
        },
    'lr_scheduler': ExponentialLR,
    'lr_scheduler_params': {
        'gamma': 0.97
    },
    'embedding': Embedding,
    'embedding_params': {
        'num_embeddings': train_dataset.num_tokens,
        'embedding_dim': 128,
        'padding_idx': train_dataset.tokens.index(' ')
    },
    'encoder': RNNEncoder,
    'encoder_params': {
        'input_size': 128,
        'layer': "LSTM",
        'encoder_dim': 128,
        'n_layers': 4,
        'dropout': 0.8,
        'is_bidirectional': False
    },
    'mlp': OpenChemMLP,
    'mlp_params': {
        'input_size': 128,
        'n_layers': 2,
        'hidden_size': [128, 1],
        'activation': [nn.ReLU(inplace=True), identity],
        'dropout': 0.0
    }
}

new_model_params = model_params.copy()
new_model_params['train_data_layer'] = None
new_model_params['val_data_layer'] = None
new_model_params['tokens'] = train_dataset.tokens
new_model_params['world_size'] = 1

import pickle
pickle.dump(new_model_params, open(model_params['logdir'] + '/params.pkl', 'wb'))

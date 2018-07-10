from models.Smiles2Label import Smiles2Label
from modules.embeddings.basic_embedding import Embedding
# from modules.encoders.rnn_encoder import RNNEncoder
from modules.encoders.cnn_encoder import CNNEncoder
from modules.mlp.openchem_mlp import OpenChemMLP
from data.utils import OpenChemDataset

import torch.nn as nn
from torch.optim import RMSprop, Adam
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
from sklearn.metrics import f1_score

train_dataset = OpenChemDataset('/home/mpopova/Work/EAGCN/Data/HIV_updated.csv',
                                cols=[0, 2])
val_dataset = OpenChemDataset('/home/mpopova/Work/EAGCN/Data/HIV_updated.csv',
                              cols=[0, 2])

use_cuda = True

model = Smiles2Label

model_params = {
    'use_cuda': use_cuda,
    'task': 'classification',
    'random_seed': 5,
    'use_clip_grad': True,
    'max_grad_norm': 10.0,
    'batch_size': 128,
    'num_epochs': 100,
    'logdir': '/home/mpopova/Work/Project/logs',
    'print_every': 1,
    'save_every': 5,
    'train_data_layer': train_dataset,
    'val_data_layer': val_dataset,
    'eval_metrics': f1_score,
    'criterion': nn.CrossEntropyLoss(),
    'optimizer': RMSprop,
    'optimizer_params': {
        'lr': 0.001
    },
    'lr_scheduler': ExponentialLR,
    'lr_scheduler_params': {
        'gamma': 0.97
    },
    'embedding': Embedding,
    'embedding_params': {
        'num_embeddings': max(train_dataset.num_tokens, val_dataset.num_tokens),
        'embedding_dim': 256,
        'padding_idx': 0
    },
    'encoder': CNNEncoder,
    'encoder_params': {
        'input_size': 256,
        'encoder_dim': 200,
        'dropout': 0.3,
        'convolutions': [(256, 7), ]*5,
    },
    'mlp': OpenChemMLP,
    'mlp_params': {
        'input_size': 400,
        'n_layers': 2,
        'hidden_size': [128, 2],
        'activations': [F.relu, F.relu],
        'dropouts': [0.3, 0.8]
    }
}

my_model = model(model_params)

if use_cuda:
    my_model = my_model.cuda()

my_model.fit()
my_model.evaluate()

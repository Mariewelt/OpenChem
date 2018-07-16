from tests.distributed_test import DistributedTest
from modules.embeddings.basic_embedding import Embedding
from modules.encoders.cnn_encoder import CNNEncoder
from modules.mlp.openchem_mlp import OpenChemMLP
from data.smiles_data_layer import SmilesDataset
import torch.nn.functional as F

import torch.nn as nn
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import f1_score

train_dataset = SmilesDataset('./benchmark_datasets/HIV_dataset/HIV_train.csv',
                              cols_to_read=[0, 1])
val_dataset = SmilesDataset('./benchmark_datasets/HIV_dataset/HIV_test.csv',
                            cols_to_read=[0, 1])

use_cuda = True

model = DistributedTest

model_params = {
    'use_cuda': use_cuda,
    'task': 'regression',
    'random_seed': 5,
    'use_clip_grad': True,
    'max_grad_norm': 10.0,
    'batch_size': 128,
    'num_epochs': 100,
    'logdir': '/home/mpopova/Work/OpenChem/logs',
    'print_every': 1,
    'save_every': 5,
    'train_data_layer': train_dataset,
    'val_data_layer': val_dataset,
    'eval_metrics': f1_score,
    'criterion': nn.MSELoss(),
    'optimizer': RMSprop,
    'optimizer_params': {
        'lr': 0.001
    },
    'lr_scheduler': ExponentialLR,
    'lr_scheduler_params': {
        'gamma': 0.97
    },
    'input_size': len(train_dataset[0]['tokenized_smiles'])
}



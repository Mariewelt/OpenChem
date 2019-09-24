"""
Unit tests for testing if an instance of every class in openchem.modules
can be created
"""
import torch

from openchem.modules.embeddings.basic_embedding import Embedding
from openchem.modules.embeddings.openchem_embedding import OpenChemEmbedding
from openchem.modules.encoders.openchem_encoder import OpenChemEncoder
from openchem.modules.encoders.rnn_encoder import RNNEncoder
from openchem.modules.encoders.gcn_encoder import GraphCNNEncoder
from openchem.modules.encoders.edge_attention_encoder import GraphEdgeAttentionEncoder
from openchem.modules.mlp.openchem_mlp import OpenChemMLP


def test_openchem_embedding_module():
    embedding_params = {'num_embeddings': 10, 'embedding_dim': 10, 'padding_idx': 0}
    embedding_layer = OpenChemEmbedding(embedding_params)


def test_basic_embedding_module():
    embedding_params = {'num_embeddings': 10, 'embedding_dim': 10, 'padding_idx': 0}
    embedding_layer = Embedding(embedding_params)


def test_openchem_encoder():
    encoder_params = {'input_size': 12, 'encoder_dim': 12}
    encoder = OpenChemEncoder(encoder_params, use_cuda=True)


def test_rnn_encoder():
    encoder_params = {'input_size': 12, 'layer': 'LSTM', 'encoder_dim': 12, 'n_layers': 1, 'is_bidirectional': False}
    encoder = RNNEncoder(encoder_params, use_cuda=True)


def test_gcn_encoder():
    encoder_params = {
        'input_size': 12,
        'encoder_dim': 10,
        'hidden_size': [10, 10, 10],
        'n_layers': 3,
    }
    encoder = GraphCNNEncoder(encoder_params, use_cuda=True)


def test_edge_attention_encoder():
    encoder_params = {
        'input_size': 12,
        'encoder_dim': 10,
        'hidden_size': [10, 10, 10],
        'n_layers': 3,
        'edge_attr_sizes': [10, 10, 10]
    }
    encoder = GraphEdgeAttentionEncoder(encoder_params, use_cuda=True)


def test_mlp():
    mlp_params = {
        'input_size': 10,
        'n_layers': 3,
        'hidden_size': [10, 10, 10],
        'activation': torch.softmax,
    }
    mlp = OpenChemMLP(mlp_params)

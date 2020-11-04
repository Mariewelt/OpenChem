:github_url: https://github.com/Mariewelt/OpenChem


GraphCNN for predicting logP
==========================

In this tutorial we will build a Graph Convolution Neural Network to solve the task of predicting partition coefficient log P. log P values are continuous, so this is a regression task.

Defining node attributes
-------------------------

We'll start with specifying atom features aka node attributes. Graph Convolution model requires user-defined function for calculating node attributes. It's a Python function that takes ``RDKit`` atom object as an input and returns dictionary of atomic attributes for this atom. Examples of node attributes are atom element type, valence, charge, hybridization, aromaticity, etc.

OpenChem also provides utilities for converting SMILES data into graphs. Check :ref:`API documentation <api-docs>`.

Here is an examples of how attributes are defined::

    def get_atomic_attributes(atom):
    attr_dict = {}

    atomic_num = atom.GetAtomicNum()
    atomic_mapping = {5: 0, 7: 1, 6: 2, 8: 3, 9: 4, 15: 5, 16: 6, 17: 7, 35: 8,
                      53: 9}
    if atomic_num in atomic_mapping.keys():
        attr_dict['atom_element'] = atomic_mapping[atomic_num]
    else:
        attr_dict['atom_element'] = 10
    attr_dict['valence'] = atom.GetTotalValence()
    attr_dict['charge'] = atom.GetFormalCharge()
    attr_dict['hybridization'] = atom.GetHybridization().real
    attr_dict['aromatic'] = int(atom.GetIsAromatic())
    return attr_dict

    node_attributes = {}
    node_attributes['valence'] = Attribute('node', 'valence', one_hot=True, values=[1, 2, 3, 4, 5, 6])
    node_attributes['charge'] = Attribute('node', 'charge', one_hot=True, values=[-1, 0, 1, 2, 3, 4])
    node_attributes['hybridization'] = Attribute('node', 'hybridization',
                                                 one_hot=True, values=[0, 1, 2, 3, 4, 5, 6, 7])
    node_attributes['aromatic'] = Attribute('node', 'aromatic', one_hot=True,
                                            values=[0, 1])
    node_attributes['atom_element'] = Attribute('node', 'atom_element',
                                                one_hot=True,
                                                values=list(range(11)))


Loading data
-------------

OpenChem provides log P dataset as a benchmark dataset, so you can load it from benchmark datasets folder with OpenChem ``read_smiles_property_file`` function::

    from openchem.data.utils load read_smiles_property_file
    data = read_smiles_property_file('./benchmark_datasets/logp_dataset/logP_labels.csv', cols_to_read=[1, 2])
    smiles = data[0]
    labels = data[1]

Now we will split data into training and test::

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(smiles, labels, test_size=0.2, random_state=42)

And save train and test splits to new files with OpenChem save_smiles_property_file utility::

    from openchem.data.utils import save_smiles_property_file
    save_smiles_property('./benchmark_datasets/logp_dataset/train.smi', X_train, y_train)
    save_smiles_property('./benchmark_datasets/logp_dataset/test.smi', X_test, y_test)

Now you can create graph data layer from input files with SMILES and labels::

    train_dataset = GraphDataset(get_atomic_attributes, node_attributes,
                                 './benchmark_datasets/logp_dataset/train.smi',
                                 delimiter=',', cols_to_read=[0, 1])
    test_dataset = GraphDataset(get_atomic_attributes, node_attributes,
                                 './benchmark_datasets/logp_dataset/test.smi',
                                 delimiter=',', cols_to_read=[0, 1])

Defining model architechture
-----------------------------

Once you created datasets, you can specify a model. We will use :class:`Graph2Label<openchem.models.Graph2Label.Graph2Label>` modality which is similar to the model described in this_ paper.

.. _this: https://pubs.acs.org/doi/abs/10.1021/acscentsci.6b00367

This model consists of 5 layers of Graph Convolutions with the size of hidden layer of 128, followed by 2 layer multilayer perceptron (MLP) with ReLU nonlinearity and hidden dimensionalities of 128 and 1. We use PyTorch Adam optimizer and MultiStepLR learning scheduler. For external evaluation we will use ``r2_score``::


    from openchem.models.Graph2Label import Graph2Label
    from openchem.modules.encoders.gcn_encoder import GraphCNNEncoder
    from openchem.modules.mlp.openchem_mlp import OpenChemMLP

    from torch.optim import Adam
    from torch.optim.lr_scheduler import StepLR
    import torch.nn.functional as F
    from sklearn.metrics import r2_score

    model = Graph2Label

    model_params = {
        'task': 'regression',
        'random_seed': 42,
        'use_clip_grad': False,
        'batch_size': 256,
        'num_epochs': 101,
        'logdir': 'logs/logp_gcnn_logs',
        'print_every': 10,
        'save_every': 5,
        'train_data_layer': train_dataset,
        'val_data_layer': test_dataset,
        'eval_metrics': r2_score,
        'criterion': nn.MSELoss(),
        'optimizer': Adam,
        'optimizer_params': {
            'lr': 0.0005,
        },
        'lr_scheduler': StepLR,
        'lr_scheduler_params': {
            'step_size': 15,
            'gamma': 0.8
        },
        'encoder': GraphCNNEncoder,
        'encoder_params': {
            'input_size': train_dataset[0]["node_feature_matrix"].shape[1],
            'encoder_dim': 128,
            'n_layers': 5,
            'hidden_size': [128]*5,
        },
        'mlp': OpenChemMLP,
        'mlp_params': {
            'input_size': 128,
            'n_layers': 2,
            'hidden_size': [128, 1],
            'activation': [F.relu, identity]
        }
    }

All of the above code should be saved in a python file. We will call it ``logP_gcnn_config.py``.

Training and evaluating the model
----------------------------------

Now as we loaded the datasets and defined the model architechture we can launch training and evaluation process from the terminal.

Suppose we have a machine with 4 GPUs, so we want to run training in distributed mode. We also want to see the evaluation metrics while the training is in progress. All the parameters from config file can be redefined in command line and parsed to the run script as arguments. So, we can, for example, change the batch size and number of epochs::

     python launch.py --nproc_per_node=4 run.py --config_file="./logP_gcnn_config.py" --mode="train_eval" --batch_size=256 --num_epochs=100


The output will be::

    Directory created
    *** Starting training from scratch process 3
    *** Starting training from scratch process 2
    *** Starting training from scratch process 0
    *** Starting training from scratch process 1
    Distributed process with rank 2 initiated
    Distributed process with rank 0 initiated
    Distributed process with rank 1 initiated
    Distributed process with rank 3 initiated
    TRAINING: [Time: 0m 2s, Epoch: 0, Progress: 0%, Loss: 6.6458]
    EVALUATION: [Time: 0m 0s, Loss: 9.3421, Metrics: -1.7553]
    TRAINING: [Time: 0m 30s, Epoch: 10, Progress: 9%, Loss: 0.6615]
    EVALUATION: [Time: 0m 0s, Loss: 0.7187, Metrics: 0.7797]
    TRAINING: [Time: 0m 59s, Epoch: 20, Progress: 19%, Loss: 0.2883]
    EVALUATION: [Time: 0m 0s, Loss: 0.3752, Metrics: 0.8838]
    TRAINING: [Time: 1m 27s, Epoch: 30, Progress: 29%, Loss: 0.2386]
    EVALUATION: [Time: 0m 0s, Loss: 0.4741, Metrics: 0.8525]
    TRAINING: [Time: 1m 56s, Epoch: 40, Progress: 39%, Loss: 0.1678]
    EVALUATION: [Time: 0m 0s, Loss: 0.3098, Metrics: 0.9036]
    TRAINING: [Time: 2m 24s, Epoch: 50, Progress: 49%, Loss: 0.1827]
    EVALUATION: [Time: 0m 0s, Loss: 0.3661, Metrics: 0.8860]
    TRAINING: [Time: 2m 54s, Epoch: 60, Progress: 59%, Loss: 0.1364]
    EVALUATION: [Time: 0m 0s, Loss: 0.3084, Metrics: 0.9044]
    TRAINING: [Time: 3m 23s, Epoch: 70, Progress: 69%, Loss: 0.1356]
    EVALUATION: [Time: 0m 0s, Loss: 0.2910, Metrics: 0.9093]
    TRAINING: [Time: 3m 51s, Epoch: 80, Progress: 79%, Loss: 0.1276]
    EVALUATION: [Time: 0m 0s, Loss: 0.3355, Metrics: 0.8959]
    TRAINING: [Time: 4m 20s, Epoch: 90, Progress: 89%, Loss: 0.1096]
    EVALUATION: [Time: 0m 0s, Loss: 0.2944, Metrics: 0.9085]
    TRAINING: [Time: 4m 50s, Epoch: 100, Progress: 99%, Loss: 0.1029]
    EVALUATION: [Time: 0m 0s, Loss: 0.3153, Metrics: 0.9020]

Model checkpoints and tensorboard log are saved to ``logdir`` folder specified in the configuration file.

Now you can evaluate model::

    python launch.py --nproc_per_node=4 run.py --config_file="./logP_gcnn_config.py" --mode="eval"

The output will be::

    *** Loading model from /home/user/OpenChem/logs/checkpoint/epoch_100
    *** Loading model from /home/user/OpenChem/logs/checkpoint/epoch_100
    *** Loading model from /home/user/OpenChem/logs/checkpoint/epoch_100
    *** Loading model from /home/user/OpenChem/logs/checkpoint/epoch_100
    Distributed process with rank 3 initiated
    Distributed process with rank 1 initiated
    Distributed process with rank 2 initiated
    Distributed process with rank 0 initiated
    => loading model  pre-trained model
    => loading model  pre-trained model
    => loading model  pre-trained model
    => loading model  pre-trained model
    EVALUATION: [Time: 0m 6s, Loss: 0.3153, Metrics: 0.9020]

So, we trained a Graph Convolution Neural Network for predicting partition coefficient logP and got test set R-squared score of 0.90.

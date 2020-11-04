:github_url: https://github.com/Mariewelt/OpenChem

Tox21 Challenge
==============================================================================

In this tutorial we will build a Recurrent model for tox21 challenge.

Loading data
-------------

Tox21 dataset is available as a benchmark dataset, so you can load it from benchmark datasets folder with OpenChem ``read_smiles_property_file`` function::

    import numpy as np
    from openchem.data.utils load read_smiles_property_file
    data = read_smiles_property_file('./benchmark_datasets/tox21/tox21.csv',
                                     cols_to_read=[13] + list(range(0,12)))
    smiles = data[0]
    labels = np.array(data[1:])

Tox21 data requires some preprocessing. As it is a multi-target dataset, some of the labels are not available and therefore just left empty. We need to fill them with dummy index, that will be ignored during training. Let's choose '999' as a dummy index::

    labels[np.where(labels=='')] = '999'
    labels = labels.T

We will also extract unique tokens from the whole dataset before splitting it into train and test in order to avoid the situation, when some of the tokens will not be present in one of the pieces of the dataset::

    from openchem.data.utils import get_tokens
    tokens, _, _ = get_tokens(smiles)
    tokens = tokens + ' '

Now we will split data into training and test::

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(smiles, labels, test_size=0.2,
                                                        random_state=42)

And save train and test splits to new files with OpenChem ``save_smiles_property_file`` utility::

    from openchem.data.utils import save_smiles_property_file
    save_smiles_property_file('./benchmark_datasets/tox21/train.smi', X_train, y_train)
    save_smiles_property_file('./benchmark_datasets/tox21/test.smi', X_test, y_test)

Now you can create SMILES data layer from input files. We will pass tokens as an argument for data layer. We will also use data augmentation by SMILES enumeration_. The idea behind it is to include non-canonical notation for SMILES. Augmentation is enabled by setting the argument ``augment=True`` when creating an object of class :class:`SmilesDataset<openchem.data.smiles_data_layer.SmilesDataset>`::

    from openchem.data.graph_data_layer import SmilesDataset
    train_dataset = SmilesDataset('./benchmark_datasets/tox21/train.smi',
                                  delimiter=',', cols_to_read=list(range(13)),
                                  tokens=tokens, augment=True)
    test_dataset = SmilesDataset('./benchmark_datasets/tox21/test.smi',
                                delimiter=',', cols_to_read=list(range(13)),
                                tokens=tokens)


.. _enumeration: https://arxiv.org/abs/1703.07076

Note that we only need to augment training dataset.

Defining evaluation function
-----------------------------

We will also need to implement our own evaluation function for calculating classification accuracy separately for each task. As an accuracy metrics we will use AUC::

    def multitask_auc(ground_truth, predicted):
    from sklearn.metrics import roc_auc_score
    import numpy as np
    ground_truth = np.array(ground_truth)
    predicted = np.array(predicted)
    n_tasks = ground_truth.shape[1]
    auc = []
    for i in range(n_tasks):
        ind = np.where(ground_truth[:, i] != 999)[0]
        auc.append(roc_auc_score(ground_truth[ind, i], predicted[ind, i]))
    return np.mean(auc)

Defining model architechture
-----------------------------

Now we define model architecture. We will use :class:`Smiles2Label<openchem.models.Smiles2Label.Smiles2Label>` modality.

This model consists of Embedding block, Recurrent Encoder with 4 LSTM layers and MLP. We will use dropout with high probability to enable regularization to avoid model overfitting::

    model = Smiles2Label

    model_params = {
        'use_cuda': True,
        'task': 'multitask',
        'random_seed': 5,
        'use_clip_grad': True,
        'max_grad_norm': 10.0,
        'batch_size': 256,
        'num_epochs': 21,
        'logdir': './logs/tox21_rnn_log',
        'print_every': 5,
        'save_every': 5,
        'train_data_layer': train_dataset,
        'val_data_layer': test_dataset,
        'predict_data_layer': predict_dataset,
        'eval_metrics': multitask_auc,
        'criterion': MultitaskLoss(ignore_index=9, n_tasks=12).cuda(),
        'optimizer': RMSprop,
        'optimizer_params': {
            'lr': 0.001,
            },
        'lr_scheduler': StepLR,
        'lr_scheduler_params': {
            'step_size': 10,
            'gamma': 0.8
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
            'hidden_size': [128, 12],
            'activation': [F.relu, torch.sigmoid],
            'dropout': 0.0
        }
    }

All of the above code should be saved in a python file. We will call it ``tox21_rnn_config.py``.

Training and evaluating the model
----------------------------------

Now as we loaded the datasets and defined the model architechture we can launch training and evaluation process from the terminal.

Suppose we have a machine with 4 GPUs, so we want to run training in distributed mode. We also want to see the evaluation metrics while the training is in progress. All the parameters from config file can be redefined in command line and parsed to the run script as arguments. So, we can, for example, change the batch size and number of epochs::

     python launch.py --nproc_per_node=4 run.py --config_file="./tox21_rnn_config.py" --mode="train_eval" --batch_size=256 --num_epochs=50


The output will be::


Model checkpoints and tensorboard log are saved to ``logdir`` folder specified in the configuration file.

Now you can evaluate model::

    python launch.py --nproc_per_node=4 run.py --config_file="./tox21_rnn_config.py" --mode="eval"

The output will be::

    *** Starting training from scratch process 3
    *** Starting training from scratch process 0
    *** Starting training from scratch process 1
    *** Starting training from scratch process 2
    Distributed process with rank 1 initiated
    Distributed process with rank 2 initiated
    Distributed process with rank 0 initiated
    Distributed process with rank 3 initiated
    TRAINING: [Time: 0m 13s, Epoch: 0, Progress: 0%, Loss: 0.3052]
    EVALUATION: [Time: 0m 0s, Loss: 0.3071, Metrics: 0.6030]
    TRAINING: [Time: 1m 18s, Epoch: 5, Progress: 16%, Loss: 0.1932]
    EVALUATION: [Time: 0m 0s, Loss: 0.1867, Metrics: 0.7948]
    TRAINING: [Time: 2m 24s, Epoch: 10, Progress: 32%, Loss: 0.1828]
    EVALUATION: [Time: 0m 0s, Loss: 0.1807, Metrics: 0.8187]
    TRAINING: [Time: 3m 30s, Epoch: 15, Progress: 48%, Loss: 0.1733]
    EVALUATION: [Time: 0m 0s, Loss: 0.1794, Metrics: 0.8296]
    TRAINING: [Time: 4m 36s, Epoch: 20, Progress: 64%, Loss: 0.1680]
    EVALUATION: [Time: 0m 0s, Loss: 0.1766, Metrics: 0.8380]
    TRAINING: [Time: 5m 43s, Epoch: 25, Progress: 80%, Loss: 0.1637]
    EVALUATION: [Time: 0m 0s, Loss: 0.1778, Metrics: 0.8352]
    TRAINING: [Time: 6m 48s, Epoch: 30, Progress: 96%, Loss: 0.1614]
    EVALUATION: [Time: 0m 0s, Loss: 0.1763, Metrics: 0.8379]


Next you can run evalutaion::

    python launch.py --nproc_per_node=4 run.py --config_file="./tox21_rnn_config.py" --mode="eval"

    *** Loading model from /home/user/OpenChem/logs/checkpoint/epoch_30
    *** Loading model from /home/user/OpenChem/logs/checkpoint/epoch_30
    *** Loading model from /home/user/OpenChem/logs/checkpoint/epoch_30
    *** Loading model from /home/user/OpenChem/logs/checkpoint/epoch_30
    Distributed process with rank 3 initiated
    Distributed process with rank 1 initiated
    Distributed process with rank 2 initiated
    Distributed process with rank 0 initiated
    => loading model  pre-trained model
    => loading model  pre-trained model
    => loading model  pre-trained model
    => loading model  pre-trained model
    EVALUATION: [Time: 0m 0s, Loss: 0.1763, Metrics: 0.8379]

So, we trained a Multi-task Recurrent Neural Network for predicting biological activity for 12 receptors from tox21 challenge with mean AUC of ~0.84.

If we want to calculate per target AUC, we will need to change the external metrics function a little bit -- for example, by just adding the print statement to print per target AUCs. So, with this model we obtain the following per target AUCs on test set:

* NR-AR 0.85
* NR-AR-LBD 0.90
* NR-AhR 0.87
* NR-Aromatase 0.84
* NR-ER 0.76
* NR-ER-LBD 0.82
* NR-PPAR-gamma 0.80
* SR-ARE 0.78
* SR-ATAD5 0.85
* SR-HSE 0.84
* SR-MMP 0.87
* SR-p53 0.86


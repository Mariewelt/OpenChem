:github_url: https://github.com/Mariewelt/OpenChem

Getting started with building models in OpenChem
==========================

In this tutorial we will cover basics of model building in OpenChem by constructing a simple
multilayer perceptron neural network for prediction logP values from molecular fingerprints. This tutorial will
cover the following point:

* Data handling (reading dataset files, splitting data into train/test)

* Specifying model hyperparmeters as a dictionary

* Running model training

* Monitoring training process wiht Tensorboard

* Evaluated of the trained model

* Running trained model for prediction on new data examples

Loading data
------------

First we need to read data from file. In this example, data is located in file ``./benchmark_datasets/logp_dataset/logP_labels.csv``.
OpenChem can process text file with multiple columns. Users can specify which columns should be read and what is the delimiter.
Important to note, that the first column is ``cols_to_read`` argument must specify column with SMILES strings.
Next columns must have labels::

    data = read_smiles_property_file('./benchmark_datasets/logp_dataset/logP_labels.csv',
                                     delimiter=",",
                                     cols_to_read=[1, 2],
                                     keep_header=False)

Variable ``data`` is a list with as many objects as columns were read from the file. ``data[0]`` contains
smiles and all the rest are labels::

    smiles = data[0]
    labels = np.array(data[1:])
    labels = labels.T

After reading the data, we can split in into train and test sets using scikit-learn utility and then
save it to new files::

    X_train, X_test, y_train, y_test = train_test_split(smiles, labels, test_size=0.2,
                                                        random_state=42)
    save_smiles_property_file('./benchmark_datasets/logp_dataset/train.smi', X_train, y_train)
    save_smiles_property_file('./benchmark_datasets/logp_dataset/test.smi', X_test, y_test)

Creating PyTorch dataset
------------------------

OpenChem has multiple utilities for creating PyTorch dataset based on the data type. In this example
we are using ``FeatureDataset`` that convert SMILES strings to vectors of features with a user-defined
function, that is passed to ``FeatureDataset`` as an argument ``get_features`` with any additional
arguments passed as a dictionary in ``get_features_args``. In this example we are using RDKit fingerprint as
features, that are calculated with function ``openchem.data.utils.get_fp``. This function accepts number of
bits in fingerprint as an additional argument ``n_bits``. Same as ``read_smiles_property_finction``
OpenChem datasets accept ``cols_to_read`` and ``delimiter`` arguments.

We are creating 3 datasets -- ``train_dataset``, ``test_dataset`` and ``predict_dataset``.
``train_dataset`` and ``test_dataset`` are used for training and evaluation respectively. In these datasets
``cols_to_read`` should contain indices for columnds with SMILES string and labels.
``predict_datasets`` will be used after training is completed to get prediction for new samples and labels
are not required. Thus, ``cols_to_read`` argument here should only contain index of column to SMILES string.
``predict_dataset`` also must have an additional argument ``return_smiles=True`` to write than to a
file with predictions::

    train_dataset = FeatureDataset(filename='./benchmark_datasets/logp_dataset/train.smi',
                                   delimiter=',', cols_to_read=[0, 1],
                                   get_features=get_fp, get_features_args={"n_bits": 2048})
    test_dataset = FeatureDataset(filename='./benchmark_datasets/logp_dataset/test.smi',
                                  delimiter=',', cols_to_read=[0, 1],
                                  get_features=get_fp, get_features_args={"n_bits": 2048})
    predict_dataset = FeatureDataset(filename='./benchmark_datasets/logp_dataset/test.smi',
                                    delimiter=',', cols_to_read=[0],
                                    get_features=get_fp, get_features_args={"n_bits": 2048},
                                    return_smiles=True)

Creating OpenChem model and specifying parameters
-------------------------------------------------

Nex step is specifying model type and model parameters. In this example we are using ``MLP2Label`` model,
which is a multilayer perceptron model, that predicts labels from feature vectors::

    model = MLP2Label

Model parameter are specified as a dictionary ``model_params``. There are some essential parameters, that
are required for every model. Such parameters are:

* ``task`` -- the problem to be solved. Can be ``classification``, ``regression``, ``multitask`` or
``graph_generation``. In this example we are building model for prediction of continuous logP values, that
is why ``task`` here is ``regression``.

* ``random_seed`` -- random seed for running the experiment. Used to enforce reproducibility of the
experiments.

* ``batch_size`` -- how many samples are included in each training batch. In this example we are using ``256``.

* ``num_epochs`` -- how many passes over the training dataset to do. In this example we are making ``101`` epochs.

* ``print_every`` -- how often intermediate training-evaluation results will be printed to standard
output and log file.

* ``save_every`` -- how often intermediate model checkpoints will be saved during training.

* ``train_data_layer`` and ``val_data_layer`` -- PyTorch datasets that are used for training and
evaluation. In this example we are using ``train_dataset`` and ``test_dataset`` objects of
``FeatureDataset`` type that were defined above.

* ``predict_data_layer`` -- also a PyTorch dataset, but this parameter is not needed if the model
won't be used for making predictions for new samples.

* ``eval_metrics`` -- a user-provided function, that is used to calculated validation metrics during
evaluation process. This function must follow scikit-learn defined signature ``fun(y_true, y_pred)``.
In this example we are using r2_ score.

.. _r2: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html

* ``criterion`` -- loss function to be optimized during the training. In this case we are using
``torch.nn.MSELoss()`` which is the mean squared error often used for regression problems.

* ``optimizer`` -- optimization algorithm to be used for model training. In this case we are using Adam_
optimizer.

.. _Adam: https://pytorch.org/docs/stable/optim.html?highlight=adam#torch.optim.Adam

* ``optimizer_params`` -- dictionary of parameters for optimization algorithms. In this case we only
specify learning rate. Full list of possible parameters can be looked up on PyTorch documentation
page for the optimization algorithm.

* ``lr_scheduler`` -- learning rate decay policy. In this case we use StepLR_.
This policy decreases the learning rate by a fixed decay factor every specified number of steps.

.. _StepLR: https://pytorch.org/docs/stable/optim.html?highlight=steplr#torch.optim.lr_scheduler.StepLR

* ``lr_scheduler_params`` -- dictionary of parameters for learning rate decay policy. Full list of
possible parameters can be looked up on PyTorch documentation page for the chosen decay policy.
In this example we decreasing the learning rate by a factor ``gamma=0.9`` every ``step_size=15`` epochs.

Next set of parameters define the model architecture. They are different from model to model.
In this example we use a multiplayer perceptron and we only need to specify a few parameters:

* ``mlp`` -- type of multilayer perceptron. OpenChem has MLP with and without Batch Normalization.

* ``mlp_params`` -- dictionary of parameters for the MLP. ``input_size`` should be equal to
the number of features in the data. In our example we are using fingerprints with ``n_bits=2048``, so
``input_size=248``. ``n_layers`` -- number of layers in MLP (we are using 4). ``hidden_size`` -- list of
dimensions for each of ``n_layers``. ``dropout`` -- probability value for dropout. If this parameter is not
specified, dropout is not used. ``activation`` -- list of activation function for each layer.

Training the model
------------------

Defined above model configurations are saved to ``logp_mlp_config.py`` file located in ``example_configs``
folder. We can now launch training process by running the following command from the command line::

    CUDA_VISIBLE_DEVICES=0 python launch.py --nproc_per_node=1 run.py --config_file=example_configs/getting_started.py  --mode="train_eval"

The output will be the following::

    Distributed process with rank 1 initalized
    Distributed process with rank 0 initalized
    Directory logs/logp_mlp_logs created
    Directory logs/logp_mlp_logs/checkpoint created
    2020-11-04 12:03:29,915 openchem INFO: Running on 2 GPUs
    2020-11-04 12:03:29,915 openchem INFO: Logging directory is set to logs/logp_mlp_logs
    2020-11-04 12:03:29,915 openchem INFO: Running with config:
    batch_size:                                       256
    logdir:                                           logs/logp_mlp_logs
    lr_scheduler_params/gamma:                        0.9
    lr_scheduler_params/step_size:                    15
    mlp_params/dropout:                               0.5
    mlp_params/input_size:                            2048
    mlp_params/n_layers:                              4
    num_epochs:                                       101
    optimizer_params/lr:                              0.001
    print_every:                                      20
    random_seed:                                      42
    save_every:                                       5
    task:                                             regression
    use_cuda:                                         True

    2020-11-04 12:03:30,109 openchem INFO: Starting training from scratch
    2020-11-04 12:03:30,109 openchem INFO: Training is set up from epoch 0
      0%|                                                                                                                         | 0/101 [00:00<?, ?it/s]
      2020-11-04 12:03:30,889 openchem.fit INFO: TRAINING: [Time: 0m 0s, Epoch: 0, Progress: 0%, Loss: 4.1647]
    INFO:openchem.fit:TRAINING: [Time: 0m 0s, Epoch: 0, Progress: 0%, Loss: 4.1647]
    2020-11-04 12:03:31,057 openchem.evaluate INFO: EVALUATION: [Time: 0m 0s, Loss: 3.8076, Metrics: -0.1291]
    INFO:openchem.evaluate:EVALUATION: [Time: 0m 0s, Loss: 3.8076, Metrics: -0.1291]                                              | 1/101 [00:00<01:34,  1.06it/s]
    2020-11-04 12:03:31,439 openchem.fit WARNING: Warning: module/MLP/layers/3/bias has zero variance (i.e. constant vector)
     20%|███████████████████████▉                                                                                                 | 20/101 [00:09<00:36,  2.20it/s]
     2020-11-04 12:03:40,331 openchem.fit INFO: TRAINING: [Time: 0m 10s, Epoch: 20, Progress: 19%, Loss: 1.0274]
    INFO:openchem.fit:TRAINING: [Time: 0m 10s, Epoch: 20, Progress: 19%, Loss: 1.0274]
    2020-11-04 12:03:40,527 openchem.evaluate INFO: EVALUATION: [Time: 0m 0s, Loss: 0.8114, Metrics: 0.7690]
    INFO:openchem.evaluate:EVALUATION: [Time: 0m 0s, Loss: 0.8114, Metrics: 0.7690]
     40%|███████████████████████████████████████████████▉                                                                         | 40/101 [00:19<00:26,  2.28it/s]
     2020-11-04 12:03:49,970 openchem.fit INFO: TRAINING: [Time: 0m 19s, Epoch: 40, Progress: 39%, Loss: 0.8870]
    INFO:openchem.fit:TRAINING: [Time: 0m 19s, Epoch: 40, Progress: 39%, Loss: 0.8870]
    2020-11-04 12:03:50,208 openchem.evaluate INFO: EVALUATION: [Time: 0m 0s, Loss: 0.7198, Metrics: 0.7955]
    INFO:openchem.evaluate:EVALUATION: [Time: 0m 0s, Loss: 0.7198, Metrics: 0.7955]
     59%|███████████████████████████████████████████████████████████████████████▉                                                 | 60/101 [00:28<00:17,  2.34it/s]
     2020-11-04 12:03:59,205 openchem.fit INFO: TRAINING: [Time: 0m 29s, Epoch: 60, Progress: 59%, Loss: 0.7898]
    INFO:openchem.fit:TRAINING: [Time: 0m 29s, Epoch: 60, Progress: 59%, Loss: 0.7898]
     60%|█████████████████████████████████████████████████████████████████████████                                                | 61/101 [00:29<00:19,  2.05it/s]
     2020-11-04 12:03:59,421 openchem.evaluate INFO: EVALUATION: [Time: 0m 0s, Loss: 0.6628, Metrics: 0.8142]
    INFO:openchem.evaluate:EVALUATION: [Time: 0m 0s, Loss: 0.6628, Metrics: 0.8142]

    INFO:openchem.fit:TRAINING: [Time: 0m 38s, Epoch: 80, Progress: 79%, Loss: 0.7267]
    2020-11-04 12:04:08,692 openchem.evaluate INFO: EVALUATION: [Time: 0m 0s, Loss: 0.6504, Metrics: 0.8179]
    INFO:openchem.evaluate:EVALUATION: [Time: 0m 0s, Loss: 0.6504, Metrics: 0.8179]
     80%|█████████████████████████████████████████████████████████████████████████████████████████████████                       | 81/101 [00:38<00:09,  2.03it/s]
    INFO:openchem.fit:TRAINING: [Time: 0m 47s, Epoch: 100, Progress: 99%, Loss: 0.6791]
    2020-11-04 12:04:17,926 openchem.evaluate INFO: EVALUATION: [Time: 0m 0s, Loss: 0.6523, Metrics: 0.8189]
    INFO:openchem.evaluate:EVALUATION: [Time: 0m 0s, Loss: 0.6523, Metrics: 0.8189]
    100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:47<00:00,  2.11it/s]

The output above shows the model configurations, overall training progress, train loss, validation loss
and validation metrics, which is an R^2 score.

To further run the trained model in ``predict`` mode to obtain predictions for new samples, the
following command should be run from the command line::

     CUDA_VISIBLE_DEVICES=0 python launch.py --nproc_per_node=1 run.py --config_file=example_configs/getting_started.py  --mode="predict"

Output will be the following::

    2020-11-04 12:15:09,379 openchem INFO: Running on 1 GPUs
    2020-11-04 12:15:09,380 openchem INFO: Logging directory is set to logs/logp_mlp_logs
    2020-11-04 12:15:09,380 openchem INFO: Running with config:
    batch_size:                                       256
    logdir:                                           logs/logp_mlp_logs
    lr_scheduler_params/gamma:                        0.9
    lr_scheduler_params/step_size:                    15
    mlp_params/dropout:                               0.5
    mlp_params/input_size:                            2048
    mlp_params/n_layers:                              4
    num_epochs:                                       101
    optimizer_params/lr:                              0.001
    print_every:                                      20
    random_seed:                                      42
    save_every:                                       5
    task:                                             regression
    use_cuda:                                         True

    2020-11-04 12:15:11,731 openchem INFO: Loading model from logs/logp_mlp_logs/checkpoint/epoch_100
    2020-11-04 12:15:13,395 openchem.predict INFO: Predictions saved to logs/logp_mlp_logs/predictions.txt
    2020-11-04 12:15:13,395 openchem.predict INFO: PREDICTION: [Time: 0m 1s, Number of samples: 2835]

This output shows model configuration, where parameters were loaded from and where predictions were saved to.

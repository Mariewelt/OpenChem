:github_url: https://github.com/Mariewelt/OpenChem

How to define and train models
===============================

In this tutorial we will discuss how to construct and run models in OpenChem
without writing any code.

Models in OpenChem are defined in Python configuration file as a dictionary of parameters.
The dictionary must contain parameters that define how to run/train/evaluate
a model as well as parameters defining model architecture. OpenChem also contains
2 Python files run.py and launch.py that handle model creation and launching
distributed processes.


Arguments for launch.py
------------------------

* ``--nproc_per_node`` --- number of processes per node. This should be equal to the number of GPUs on the node.


Arguments for run.py
---------------------

* ``--config_gile`` --- path to python configuration file where model is defined.

* ``--mode`` --- "train", "train_eval" or "eval".

* ``--continue_learning`` --- if this argument is specified, training will be resumed from the latest checkpoint.

Configuration file
-------------------------

Configuration file must contain ``model``, which should be any class derived from :class:`OpenChemModel<openchem.models.openchem_model>` and ``dictionary`` ``model_params``.

Below is description of common parameters for all models that are not related to model architechture:

* task --- ``string``, specifies the task to be solved by the model. Could be ``classification``, ``regression`` or ``multitask``.

* train_data_layer --- pytorch dataset for training data. Could be ``None`` if ``--mode=eval``. OpenChem currently provides utilities for creating SMILES, Graph and MoleculeProtein datasets.

* val_data_layer --- pytorch dataset for validation data. Could be ``None`` of ``--mode=train``.

* print_every --- ``int``, how often logs will be printed.

* save_every --- ``int``, how often model will be saved to checkpoints.

* logdir --- ``string``, path to folder where model checkpoints and tensorboard log will be saved.

* use_clip_grad --- ``bool``, whether to use gradient clipping.

* max_grad_norm --- ``float``, maximum norm of parameters, if gradient clipping is used.

* batch_size --- ``int``, batch size.

* num_epochs --- ``int``, number of epochs for training. Could be ``None`` if ``--mode=eval``.

* eval_metrics --- user defined function, metrics for evaluation. Could be ``None`` if ``--mode=train``. Python scikit learn package contains majority of the evaluation metrics you would probably like to use.

* criterion --- pytorch loss, model loss.

* optimizer --- pytorch optimizer, optimizer for training the model. Could be ``None`` if ``--mode=eval``


Other parameters are specific to model architecture. Check out :ref:`API documentation <api-docs>` and other tutorials.

Launching jobs
---------------

Here is an example of job, that will be run on a node with 4 GPUs::

    python launch.py --nproc_per_node=4 run.py --config_file="./my_config.py" --mode="train"


If you want to use only specific GPUs and not all, you can use flag ``CUDA_VISIBLE_DEVICES`` and set ``--nproc_per_node`` to the number of GPUs you want to use.

For example, here is how to run a job on two GPUs with ``ids`` 0 and 1::

    CUDA_VISIBLE_DEVICES=0,1 python launch.py --nproc_per_node=2 run.py --config_file="./my_config.py" --mode="train"

If you don't want to run model in distributed mode, but instead want just run a single process on one GPU, you can use flag ``CUDA_VISIBLE_DEVICES`` and set ``--nproc_per_node=1``::

    CUDA_VISIBLE_DEVICES=0 python launch.py --nproc_per_node=1 run.py --config_file="./my_config.py" --mode="train"
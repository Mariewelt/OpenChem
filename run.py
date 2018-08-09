# modified from https://github.com/NVIDIA/OpenSeq2Seq/blob/master/run.py

import os
import ast
import copy
import runpy
import argparse

from six import string_types

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel, DataParallel

from openchem.models.openchem_model import build_training, fit, evaluate

from openchem.data.utils import create_loader
from openchem.utils.utils import get_latest_checkpoint, deco_print
from openchem.utils.utils import flatten_dict, nested_update, nest_dict


def main():
    parser = argparse.ArgumentParser(description='Experiment parameters')
    parser.add_argument("--use_cuda", default=torch.cuda.is_available(),
                        help="Whether to train on GPU")
    parser.add_argument("--config_file", required=True,
                        help="Path to the configuration file")
    parser.add_argument("--mode", default='train',
                        help="Could be \"train\", \"eval\", "
                             "\"train_eval\" or \"infer\"")
    parser.add_argument('--continue_learning', dest='continue_learning',
                        action='store_true',
                        help="whether to continue learning")
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument("--local_rank", type=int)

    args, unknown = parser.parse_known_args()

    if args.mode not in ['train', 'eval', 'train_eval', 'infer']:
        raise ValueError("Mode has to be one of "
                         "['train', 'eval', 'train_eval', 'infer']")
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    config_module = runpy.run_path(args.config_file)

    model_config = config_module.get('model_params', None)
    model_config['use_cuda'] = args.use_cuda
    if model_config is None:
        raise ValueError('model_params dictionary has to be '
                         'defined in the config file')
    model_object = config_module.get('model', None)
    if model_object is None:
        raise ValueError('model class has to be defined in the config file')

        # after we read the config, trying to overwrite some of the properties
        # with command line arguments that were passed to the script
    parser_unk = argparse.ArgumentParser()
    for pm, value in flatten_dict(model_config).items():
        if type(value) == int or type(value) == float or \
                isinstance(value, string_types):
            parser_unk.add_argument('--' + pm, default=value, type=type(value))
        elif type(value) == bool:
            parser_unk.add_argument('--' + pm, default=value,
                                    type=ast.literal_eval)

    config_update = parser_unk.parse_args(unknown)
    nested_update(model_config, nest_dict(vars(config_update)))

    # checking that everything is correct with log directory
    logdir = model_config['logdir']
    ckpt_dir = logdir + '/checkpoint/'

    if args.local_rank == 0 or args.local_rank == -1:
        try:
            try:
                os.stat(logdir)
            except:
                os.mkdir(logdir)
                print('Directory created')
            # check if folder checkpoint within log directory exists,
            # create if it doesn't
            try:
                os.stat(ckpt_dir)
            except:
                os.mkdir(logdir + '/checkpoint')
                print("Directory created")
                ckpt_dir = logdir + '/checkpoint/'
            if args.mode == 'train' or args.mode == 'train_eval':
                if os.path.isfile(logdir):
                    raise IOError(
                        "There is a file with the same name as \"logdir\" "
                        "parameter. You should change the log directory path "
                        "or delete the file to continue.")

                # check if 'logdir' directory exists and non-empty
                if os.path.isdir(ckpt_dir) and os.listdir(ckpt_dir) != []:
                    if not args.continue_learning:
                        raise IOError(
                            "Log directory is not empty. If you want to "
                            "continue learning, you should provide "
                            "\"--continue_learning\" flag")
                    checkpoint = get_latest_checkpoint(ckpt_dir)
                    if checkpoint is None:
                        raise IOError(
                            "There is no model checkpoint in the "
                            "{} directory. Can't load model".format(ckpt_dir)
                        )
                else:
                    if args.continue_learning:
                        raise IOError(
                            "The log directory is empty or does not exist. "
                            "You should probably not provide "
                            "\"--continue_learning\" flag?")
                    checkpoint = None
            elif args.mode == 'eval' or args.mode == 'infer':
                if os.path.isdir(logdir) and os.listdir(logdir) != []:
                    checkpoint = get_latest_checkpoint(ckpt_dir)
                    if checkpoint is None:
                        raise IOError(
                                "There is no model checkpoint in the "
                                "{} directory. Can't load model".format(
                                    ckpt_dir
                                )
                        )
                else:
                    raise IOError(
                        "{} does not exist or is empty, can't restore".format(
                            ckpt_dir
                        )
                    )
        except IOError:
                raise
    else:
        if args.continue_learning or args.mode == 'eval':
            checkpoint = get_latest_checkpoint(ckpt_dir)
        else:
            checkpoint = None

    train_config = copy.deepcopy(model_config)
    eval_config = copy.deepcopy(model_config)

    args.distributed = args.local_rank >= 0

    if args.mode == 'train' or args.mode == 'train_eval':
        if 'train_params' in config_module:
            nested_update(train_config,
                          copy.deepcopy(config_module['train_params']))
    if args.mode == 'eval' or args.mode == 'train_eval' or args.mode == 'infer':
        if 'eval_params' in config_module:
            nested_update(eval_config,
                          copy.deepcopy(config_module['eval_params']))

    if args.mode == 'train' or args.mode == 'train_eval':
        if not args.continue_learning:
            if args.distributed:
                deco_print("Starting training from scratch process " +
                           str(args.local_rank))
            else:
                deco_print("Starting training from scratch")
        else:
            deco_print(
                "Restored checkpoint from {}. Resuming training".format(
                    checkpoint),
            )
    elif args.mode == 'eval' or args.mode == 'infer':
        deco_print("Loading model from {}".format(checkpoint))

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend=args.dist_backend,
                                init_method='env://')
        print('Distributed process with rank ' + str(args.local_rank) +
              ' initiated')

        args.world_size = torch.distributed.get_world_size()
        model_config['world_size'] = args.world_size
    else:
        model_config['world_size'] = 1

    cudnn.benchmark = True

    if args.mode == "train" or args.mode == "train_eval":
        train_dataset = copy.deepcopy(model_config['train_data_layer'])
        if args.distributed:
            train_sampler = DistributedSampler(train_dataset)
        else:
            train_sampler = None
        train_loader = create_loader(train_dataset,
                                     batch_size=model_config['batch_size'],
                                     shuffle=(train_sampler is None),
                                     num_workers=args.workers,
                                     pin_memory=True,
                                     sampler=train_sampler)
    else:
        train_loader = None

    if args.mode in ["eval", "train_eval"]:
        val_dataset = copy.deepcopy(model_config['val_data_layer'])
        val_loader = create_loader(val_dataset,
                                   batch_size=model_config['batch_size'],
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=True)
    else:
        val_loader = None

    model_config['train_loader'] = train_loader
    model_config['val_loader'] = val_loader

    # create model
    model = model_object(params=model_config)

    model = model.cuda()

    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank],
                                        output_device=args.local_rank
                                        )
    else:
        model = DataParallel(model)
    if args.continue_learning or args.mode == 'eval':
        print("=> loading model  pre-trained model")
        weights = torch.load(checkpoint)
        model.load_state_dict(weights)

    criterion, optimizer, lr_scheduler = build_training(model, model_config)

    if args.mode == 'train':
        fit(model, lr_scheduler, train_loader, optimizer, criterion,
            model_config, eval=False)
    elif args.mode == 'train_eval':
        fit(model, lr_scheduler, train_loader, optimizer, criterion,
            model_config, eval=True, val_loader=val_loader)
    elif args.mode == "eval":
        evaluate(model, val_loader, criterion)


if __name__ == '__main__':
    main()


# modified from https://github.com/NVIDIA/OpenSeq2Seq/blob/master/run.py

import os
import ast
import copy
import runpy
import warnings
import argparse

from six import string_types

import torch
import torch.utils.data as data
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler

from data.utils import create_loader
from utils.utils import get_latest_checkpoint, deco_print
from utils.utils import flatten_dict, nested_update, nest_dict


def main():
    parser = argparse.ArgumentParser(description='Experiment parameters')
    parser.add_argument("--use_cuda", default=torch.cuda.is_available(),
                        help="Whether to train on GPU")
    parser.add_argument("--config_file", required=True,
                        help="Path to the configuration file")
    parser.add_argument("--mode", default='train',
                        help="Could be \"train\", \"eval\", or  "
                             "\"train_eval\"")
    parser.add_argument('--continue_learning', dest='continue_learning',
                        action='store_true',
                        help="whether to continue learning")
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=str,
                        help='GPU id to use.')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument("--local_rank", type=int)

    args, unknown = parser.parse_known_args()

    if args.gpu is not None:
        args.gpu = args.gpu.split(',')
        args.gpu = [int(x) for x in args.gpu]

    if args.mode not in ['train', 'eval', 'train_eval']:
        raise ValueError("Mode has to be one of "
                         "['train', 'eval', 'train_eval']")

    config_module = runpy.run_path(args.config_file)

    model_config = config_module.get('model_params', None)
    model_config['use_cuda'] = args.use_cuda
    if model_config is None:
        raise ValueError('model_params dictionary has to be '
                         'defined in the config file')
    model = config_module.get('model', None)
    if model is None:
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
            ckpt_dir = logdir + '/checkpoint'
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
                        "Log directory is not empty. If you want to continue "
                        "learning, you should provide "
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
        elif args.mode == 'eval':
            if os.path.isdir(logdir) and os.listdir(logdir) != []:
                checkpoint = get_latest_checkpoint(ckpt_dir)
                if checkpoint is None:
                    raise IOError(
                            "There is no model checkpoint in the "
                            "{} directory. Can't load model".format(ckpt_dir)
                    )
            else:
                raise IOError(
                    "{} does not exist or is empty, can't restore model".format(
                        ckpt_dir
                    )
                )
    except IOError:
            raise

    train_config = copy.deepcopy(model_config)
    eval_config = copy.deepcopy(model_config)

    if args.mode == 'train' or args.mode == 'train_eval':
        if 'train_params' in config_module:
            nested_update(train_config,
                          copy.deepcopy(config_module['train_params']))
    if args.mode == 'eval' or args.mode == 'train_eval':
        if 'eval_params' in config_module:
            nested_update(eval_config,
                          copy.deepcopy(config_module['eval_params']))

    if args.mode == 'train' or args.mode == 'train_eval':
        if checkpoint is None:
            deco_print("Starting training from scratch")
        else:
            deco_print(
                "Restored checkpoint from {}. Resuming training".format(
                    checkpoint),
            )
    elif args.mode == 'eval' or args.mode == 'infer':
        deco_print("Loading model from {}".format(checkpoint))

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend,
                                     init_method=args.dist_url)
        # dist.init_process_group(backend=args.dist_backend,
        #                         init_method=args.dist_url,
        #                         world_size=args.world_size)
        print('Distirbuted process initiated')

    if args.gpu is not None and len(args.gpu) == 1:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

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
        if args.distributed:
            val_sampler = DistributedSampler(val_dataset)
        else:
            val_sampler = None
        val_loader = create_loader(val_dataset,
                                   batch_size=model_config['batch_size'],
                                   shuffle=(train_sampler is None),
                                   num_workers=args.workers,
                                   pin_memory=True,
                                   sampler=val_sampler)
    else:
        val_loader = None

    model_config['train_loader'] = train_loader
    model_config['val_loader'] = val_loader

    # create model
    if args.continue_learning or args.mode == 'eval':
        print("=> loading model  pre-trained model")
        my_model = model(params=model_config)
        my_model.load_model(ckpt_dir)
    else:
        print("=> creating model")
        my_model = model(params=model_config)

    if args.gpu is not None and len(args.gpu) == 1:
        my_model = model.cuda(args.gpu)
    elif args.distributed and args.gpu is None:
        my_model = torch.nn.parallel.DistributedDataParallel(my_model,
							     device_ids=[args.local_rank],
                                                             output_device=args.local_rank
                                                             ).cuda()
    elif args.distributed and len(args.gpu) > 1:
        my_model = torch.nn.parallel.DistributedDataParallel(my_model,
                                                             device_ids=[args.local_rank],
							     output_device=args.local_rank
                                                             ).cuda()
    elif args.use_cuda:
        my_model = torch.nn.DataParallel(my_model, device_ids=args.gpu).cuda()

    if args.mode == 'train':
        my_model.module.fit(eval=False)
    elif args.mode == 'train_eval':
        my_model.module.fit(eval=True)
    elif args.mode == "eval":
        my_model.module.evaluate()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn")
    main()

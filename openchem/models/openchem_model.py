import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist

from openchem.utils.utils import check_params

import time

from openchem.utils.logger import Logger
from openchem.utils.utils import time_since, calculate_metrics
from openchem.optimizer.openchem_optimizer import OpenChemOptimizer
from openchem.optimizer.openchem_lr_scheduler import OpenChemLRScheduler

import numpy as np


class OpenChemModel(nn.Module):
    """Base class for all OpenChem models. Function :func:'forward' and
    :func:'cast' inputs must be overridden for every class, that inherits from
    OpenChemModel.
    """
    def __init__(self, params):
        super(OpenChemModel, self).__init__()
        check_params(params, self.get_required_params(),
                     self.get_optional_params())
        self.params = params
        self.use_cuda = self.params['use_cuda']
        self.batch_size = self.params['batch_size']
        self.eval_metrics = self.params['eval_metrics']
        self.task = self.params['task']
        self.logdir = self.params['logdir']
        self.world_size = self.params['world_size']

        self.num_epochs = self.params['num_epochs']
        self.use_clip_grad = self.params['use_clip_grad']
        if self.use_clip_grad:
            self.max_grad_norm = self.params['max_grad_norm']
        else:
            self.max_grad_norm = None
        self.random_seed = self.params['random_seed']
        self.print_every = self.params['print_every']
        self.save_every = self.params['save_every']

    @staticmethod
    def get_required_params():
        return{
            'task': str,
            'batch_size': int,
            'num_epochs': int,
            'train_data_layer': None,
            'val_data_layer': None,
        }

    @staticmethod
    def get_optional_params():
        return{
            'use_cuda': bool,
            'use_clip_grad': bool,
            'max_grad_norm': float,
            'random_seed': int,
            'print_every': int,
            'save_every': int,
            'lr_scheduler': None,
            'lr_scheduler_params': dict,
            'eval_metrics': None,
            'logdir': str
        }

    def forward(self, inp, eval=False):
        raise NotImplementedError

    def cast_inputs(self, sample):
        raise NotImplementedError

    def load_model(self, path):
        weights = torch.load(path)
        self.load_state_dict(weights)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

        
def build_training(model, params):

    optimizer = OpenChemOptimizer([params['optimizer'],
                                   params['optimizer_params']],
                                  model.parameters())
    lr_scheduler = OpenChemLRScheduler([params['lr_scheduler'],
                                        params['lr_scheduler_params']],
                                       optimizer.optimizer)
    use_cuda = params['use_cuda']
    criterion = params['criterion']
    if use_cuda:
        criterion = criterion.cuda()
    # train_loader = params['train_loader']
    # val_loader = params['val_loader']
    return criterion, optimizer, lr_scheduler #, train_loader, val_loader


def train_step(model, optimizer, criterion, inp, target):
    optimizer.zero_grad()
    output = model.forward(inp, eval=False)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    has_module = False
    if hasattr(model, 'module'):
        has_module = True
    if has_module:
        use_clip_grad = model.module.use_clip_grad
        max_grad_norm = model.module.max_grad_norm
    else:
        use_clip_grad = model.use_clip_grad
        max_grad_norm = model.max_grad_norm
    if use_clip_grad:
        clip_grad_norm_(model.parameters(), max_grad_norm)

    return loss


def print_logs(world_size):
    if world_size == 1:
        return True
    elif torch.distributed.get_rank() == 0:
        return True
    else:
        return False


def fit(model, scheduler, train_loader, optimizer, criterion, params,
        eval=False, val_loader=None):
    cur_epoch = 0
    logdir = params['logdir']
    print_every = params['print_every']
    save_every = params['save_every']
    n_epochs = params['num_epochs']
    logger = Logger(logdir + '/tensorboard_log/')
    start = time.time()
    loss_total = 0
    n_batches = 0
    scheduler = scheduler.scheduler
    all_losses = []
    val_losses = []
    has_module = False
    if hasattr(model, 'module'):
        has_module = True
    if has_module:
        world_size = model.module.world_size
    else:
        world_size = model.world_size     

    for epoch in range(cur_epoch, n_epochs + cur_epoch):
        for i_batch, sample_batched in enumerate(train_loader):
            if has_module:
                batch_input, batch_target = model.module.cast_inputs(sample_batched)
            else:
                batch_input, batch_target = model.cast_inputs(sample_batched)
            loss = train_step(model, optimizer, criterion,
                              batch_input, batch_target)
            if world_size > 1:
                reduced_loss = reduce_tensor(loss, world_size)
            else:
                reduced_loss = loss.clone()
            loss_total += reduced_loss.item()
            n_batches += 1
        cur_loss = loss_total / n_batches
        all_losses.append(cur_loss)

        if epoch % print_every == 0:
            if print_logs(world_size):
                print('TRAINING: [Time: %s, Epoch: %d, Progress: %d%%, '
                      'Loss: %.4f]' % (time_since(start), epoch,
                                       epoch / n_epochs * 100, cur_loss))
            if eval:
                assert val_loader is not None
                val_loss, metrics = evaluate(model, val_loader, criterion)
                val_losses.append(val_loss)
                info = {'Train loss': cur_loss, 'Validation loss': val_loss,
                        'Validation metrics': metrics,
                        'LR': optimizer.param_groups[0]['lr']}
            else:
                info = {'Train loss': cur_loss,
                        'LR': optimizer.param_groups[0]['lr']}

            if print_logs(world_size):
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, epoch + 1)

                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    try:
                        logger.histo_summary(tag, value.detach().cpu().numpy(),
                                         epoch + 1)
                        logger.histo_summary(tag + '/grad',
                                         value.grad.detach().cpu().numpy(),
                                         epoch + 1)
                    except:
                        pass

        if epoch % save_every == 0 and print_logs(world_size):
            torch.save(model.state_dict(), logdir + '/checkpoint/epoch_' + str(epoch))

        loss_total = 0
        n_batches = 0
        scheduler.step()

    return all_losses, val_losses


def evaluate(model, val_loader, criterion):
    loss_total = 0
    n_batches = 0
    start = time.time()
    prediction = []
    ground_truth = []
    has_module = False
    if hasattr(model, 'module'):
        has_module = True
    if has_module:
        task = model.module.task
        eval_metrics = model.module.eval_metrics
        world_size = model.module.world_size
    else:
        task = model.task
        eval_metrics = model.eval_metrics
        world_size = model.world_size
    for i_batch, sample_batched in enumerate(val_loader):
        if has_module:
            batch_input, batch_target = model.module.cast_inputs(sample_batched)
        else:
            batch_input, batch_target = model.cast_inputs(sample_batched)
        predicted = model.forward(batch_input, eval=True)
        prediction += list(predicted.detach().cpu().numpy())
        ground_truth += list(batch_target.cpu().numpy())
        loss = criterion(predicted, batch_target)
        loss_total += loss.item()
        n_batches += 1

    cur_loss = loss_total / n_batches
    if task == 'classification':
        prediction = np.argmax(prediction, axis=1)
    metrics = calculate_metrics(prediction, ground_truth,
                                eval_metrics)
    if print_logs(world_size):
        print('EVALUATION: [Time: %s, Loss: %.4f, Metrics: %.4f]' %
              (time_since(start), cur_loss, metrics))
    return cur_loss, metrics


def reduce_tensor(tensor, world_size):
    r"""
    Reduces input ''tensor'' across all processes in such a way that everyone
    gets the sum of ''tensor'' from all of the processes.
    Args:
        tensor (Tensor): data to be reduced.
        world_size (int): number of processes.
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt

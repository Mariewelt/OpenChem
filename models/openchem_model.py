import torch
from torch import nn
from torch.nn.utils import clip_grad_norm, clip_grad_norm_

from utils.utils import check_params

import os
import time

from utils.logger import Logger
from utils.utils import time_since, calculate_metrics

import numpy as np


class OpenChemModel(nn.Module):
    """Base class for OpenChem models"""
    def __init__(self, params):
        super(OpenChemModel, self).__init__()
        check_params(params, self.get_required_params(),
                     self.get_optional_params())
        self.params = params
        self.use_cuda = self.params['use_cuda']
        self.batch_size = self.params['batch_size']
        self.train_loader = self.params['train_loader']
        self.val_loader = self.params['val_loader']
        self.eval_metrics = self.params['eval_metrics']
        self.task = self.params['task']
        self.criterion = self.params['criterion']
        if self.use_cuda:
            self.criterion = self.criterion.cuda()
        self.optimizer = None
        self.lr_scheduler = self.params['lr_scheduler']
        self.logdir = self.params['logdir']
        self._flat_grads = None
        self.distributed_world_size = self.params['world_size']

        # check if log directory exists, create if it doesn't
        # try:
        #    os.stat(self.logdir)
        # except UserWarning(self.logdir + ' directory not found. '
        #                                  'Creating...'):
        #     os.mkdir(self.logdir)
        #     print('Directory created')
        # check if folder checkpoint within log directory exists,
        # create if it doesn't
        # try:
        #    os.stat(self.logdir + '/checkpoint')
        # except UserWarning(self.logdir + '/checkpoint directory not found. '
        #                                  'Creating...'):
        #     os.mkdir(self.logdir + '/checkpoint')
        #     print('Directory created')

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
    def get_params():
        return {
            'use_cuda': bool,
            'task': str,  # could be classification, regression, multi-task
            'random_seed': int,
            'use_clip_grad': bool,
            'max_grad_norm': float,
            'batch_size': int,
            'num_epochs': int,
            'logdir': str,
            'print_every': int,
            'save_every': int,
            'train_loader': None,  # could be any user defined class
            'val_loader': None,  # could be any user defined class
            'criterion': None,  # any valid PyTorch loss function
            'optimizer': None,  # any valid PyTorch optimizer
            'optimizer_params': dict,
            'lr_scheduler': None,  # any valid PyTorch optimizer scheduler
            'lr_scheduler_params': dict,
            'eval_metrics': None  # any function specified by user
        }

    @staticmethod
    def get_required_params():
        return{
            'task': str,
            'batch_size': int,
            'num_epochs': int,
            'train_data_layer': None,
            'val_data_layer': None,
            'criterion': None,
            'optimizer': None,
            'optimizer_params': dict,
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

    def train_step(self, inp, target, multiprocess=False):
        self.optimizer.zero_grad()
        output = self.forward(inp, eval=False)
        loss = self.criterion(output, target)
        loss.backward()
        if multiprocess:
            try:
                self._all_reduce_and_rescale()
            except OverflowError as e:
                self.zero_grad()
                print('| WARNING: overflow detected, ' + str(e))
        else:
            self.optimizer.step()
            if self.use_clip_grad:
                clip_grad_norm(self.parameters(), self.max_grad_norm)

        return loss.item()

    def fit(self, eval=True, multiprocess=False):
        cur_epoch = 0
        logdir = self.logdir
        print_every = self.print_every
        save_every = self.save_every
        n_epochs = self.num_epochs
        logger = Logger(logdir + '/tensorboard_log/')
        start = time.time()
        loss_total = 0
        n_batches = 0
        scheduler = self.scheduler.scheduler
        all_losses = []
        val_losses = []

        for epoch in range(cur_epoch, n_epochs + cur_epoch):
            for i_batch, sample_batched in enumerate(self.train_loader):
                batch_input, batch_target = self.cast_inputs(sample_batched)
                loss = self.train_step(batch_input, batch_target, multiprocess)
                loss_total += loss
                n_batches += 1
            cur_loss = loss_total / n_batches
            all_losses.append(cur_loss)

            if epoch % print_every == 0:
                print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch,
                                               epoch / n_epochs * 100,
                                               cur_loss))
                if eval:
                    val_loss, metrics = self.evaluate()
                    val_losses.append(val_loss)
                    info = {'Train loss': cur_loss, 'Validation loss': val_loss,
                            'Validation metrics': metrics,
                            'LR': self.optimizer.param_groups[0]['lr']}
                else:
                    info = {'Train loss': cur_loss,
                            'LR': self.optimizer.param_groups[0]['lr']}

                for tag, value in info.items():
                    logger.scalar_summary(tag, value, epoch + 1)

                for tag, value in self.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, value.detach().cpu().numpy(),
                                         epoch + 1)
                    logger.histo_summary(tag + '/grad',
                                         value.grad.detach().cpu().numpy(),
                                         epoch + 1)
            if epoch % save_every == 0:
                torch.save(self.state_dict(), logdir + '/checkpoint/epoch_' +
                           str(epoch))

            loss_total = 0
            n_batches = 0
            scheduler.step()

        return all_losses, val_losses

    def evaluate(self):
        loss_total = 0
        n_batches = 0
        start = time.time()
        prediction = []
        ground_truth = []
        for i_batch, sample_batched in enumerate(self.val_loader):
            batch_input, batch_target = self.cast_inputs(sample_batched)
            predicted = self.forward(batch_input, eval=True)
            prediction += list(predicted.detach().cpu().numpy())
            ground_truth += list(batch_target.cpu().numpy())
            loss = self.criterion(predicted, batch_target)
            loss_total += loss.item()
            n_batches += 1

        cur_loss = loss_total / n_batches
        if self.task == 'classification':
            prediction = np.argmax(prediction, axis=1)
        metrics = calculate_metrics(prediction, ground_truth, self.eval_metrics)
        print('[%s %.4f  %.4f]' % (time_since(start), cur_loss, metrics))

        return cur_loss, metrics

    def load_model(self, path):
        weights = torch.load(path)
        self.load_state_dict(weights)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def _all_reduce_and_rescale(self):
        # modified from fairseq
        # flatten grads into a single buffer and all-reduce
        flat_grads = self._flat_grads = self._get_flat_grads(
            self._flat_grads)
        if self.distributed_world_size > 1:
            torch.distributed.all_reduce(flat_grads)

        # rescale and clip gradients
        if self.use_clip_grad:
            clip_grad_norm_(flat_grads, self.max_grad_norm)

        # copy grads back into model parameters
        self._set_flat_grads(flat_grads)

    def _get_grads(self):
        grads = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                raise RuntimeError(
                    'Model parameter did not receive gradient: ' + name + '. '
                    'Use the param in the forward pass or set '
                    'requires_grad=False')
            grads.append(p.grad.data)
        return grads

    def _get_flat_grads(self, out=None):
        grads = self._get_grads()
        if out is None:
            grads_size = sum(g.numel() for g in grads)
            out = grads[0].new(grads_size).zero_()
        offset = 0
        for g in grads:
            numel = g.numel()
            out[offset:offset + numel].copy_(g.view(-1))
            offset += numel
        return out[:offset]

    def _set_flat_grads(self, new_grads):
        grads = self._get_grads()
        offset = 0
        for g in grads:
            numel = g.numel()
            g.copy_(new_grads[offset:offset + numel].view_as(g))

        offset += numel

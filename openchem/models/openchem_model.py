import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
import logging

from openchem.utils.utils import check_params

import time
from tqdm import tqdm

from openchem.utils import comm
from tensorboardX import SummaryWriter
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
        check_params(params, self.get_required_params(), self.get_optional_params())
        self.params = params
        self.use_cuda = self.params['use_cuda']
        self.batch_size = self.params['batch_size']
        self.eval_metrics = self.params['eval_metrics']
        self.task = self.params['task']
        self.logdir = self.params['logdir']

        self.num_epochs = self.params['num_epochs']
        if 'use_clip_grad' in self.params.keys():
            self.use_clip_grad = self.params['use_clip_grad']
        else:
            self.use_clip_grad = False
        if self.use_clip_grad:
            self.max_grad_norm = self.params['max_grad_norm']
        else:
            self.max_grad_norm = None
        self.random_seed = self.params['random_seed']
        self.print_every = self.params['print_every']
        self.save_every = self.params['save_every']

    @staticmethod
    def get_required_params():
        return {
            'task': str,
            'batch_size': int,
            'num_epochs': int,
            'train_data_layer': None,
            'val_data_layer': None,
        }

    @staticmethod
    def get_optional_params():
        return {
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

    @staticmethod
    def cast_inputs(sample, task, use_cuda):
        raise NotImplementedError

    def load_model(self, path):
        weights = torch.load(path)
        self.load_state_dict(weights)

    def save_model(self, path):
        torch.save(self.state_dict(), path)


def build_training(model, params):

    optimizer = OpenChemOptimizer([params['optimizer'], params['optimizer_params']], model.parameters())
    lr_scheduler = OpenChemLRScheduler([params['lr_scheduler'], params['lr_scheduler_params']], optimizer.optimizer)
    use_cuda = params['use_cuda']
    criterion = params['criterion']
    if use_cuda:
        criterion = criterion.cuda()
    return criterion, optimizer, lr_scheduler


def train_step(model, optimizer, criterion, inp, target):
    with torch.autograd.set_detect_anomaly(True):
        optimizer.zero_grad()
        output = model(inp, eval=False)
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


def fit(model, scheduler, train_loader, optimizer, criterion, params, eval=False, val_loader=None, cur_epoch=0):
    textlogger = logging.getLogger("openchem.fit")
    logdir = params['logdir']
    print_every = params['print_every']
    save_every = params['save_every']
    n_epochs = params['num_epochs']
    writer = SummaryWriter()
    start = time.time()
    loss_total = 0
    n_batches = 0
    schedule_by_iter = scheduler.by_iteration
    scheduler = scheduler.scheduler
    all_losses = []
    val_losses = []
    has_module = False
    if hasattr(model, 'module'):
        has_module = True
    world_size = comm.get_world_size()

    for epoch in tqdm(range(cur_epoch, n_epochs + cur_epoch)):
        model.train()
        for i_batch, sample_batched in enumerate(train_loader):

            if has_module:
                task = model.module.task
                use_cuda = model.module.use_cuda
                batch_input, batch_target = model.module.cast_inputs(sample_batched, task, use_cuda)
            else:
                task = model.task
                use_cuda = model.use_cuda
                batch_input, batch_target = model.cast_inputs(sample_batched, task, use_cuda)
            loss = train_step(model, optimizer, criterion, batch_input, batch_target)
            if schedule_by_iter:
                # steps are in iters
                scheduler.step()
            if world_size > 1:
                reduced_loss = reduce_tensor(loss, world_size).item()
            else:
                reduced_loss = loss.item()
            loss_total += reduced_loss
            n_batches += 1
        cur_loss = loss_total / n_batches
        all_losses.append(cur_loss)

        if epoch % print_every == 0:
            if comm.is_main_process():
                textlogger.info('TRAINING: [Time: %s, Epoch: %d, Progress: %d%%, '
                                'Loss: %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, cur_loss))
            if eval:
                assert val_loader is not None
                val_loss, metrics = evaluate(model, val_loader, criterion, epoch=epoch)
                val_losses.append(val_loss)
                info = {
                    'Train loss': cur_loss,
                    'Validation loss': val_loss,
                    'Validation metrics': metrics,
                    'LR': optimizer.param_groups[0]['lr']
                }
            else:
                info = {'Train loss': cur_loss, 'LR': optimizer.param_groups[0]['lr']}

            if comm.is_main_process():
                for tag, value in info.items():
                    writer.add_scalar(tag, value, epoch + 1)

                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    if torch.std(value).item() < 1e-3 or \
                            torch.isnan(torch.std(value)).item():
                        textlogger.warning("Warning: {} has zero variance ".format(tag) + "(i.e. constant vector)")
                    else:
                        log_value = value.detach().cpu().numpy()
                        writer.add_histogram(tag, log_value, epoch + 1)
                        #logger.histo_summary(
                        #    tag, log_value, epoch + 1)
                        if value.grad is None:
                            print("Warning: {} grad is undefined".format(tag))
                        else:
                            log_value_grad = value.grad.detach().cpu().numpy()
                            writer.add_histogram(tag + "/grad", log_value_grad, epoch + 1)

        if epoch % save_every == 0 and comm.is_main_process():
            torch.save(model.state_dict(), logdir + '/checkpoint/epoch_' + str(epoch))

        loss_total = 0
        n_batches = 0
        if not schedule_by_iter:
            # steps are in epochs
            scheduler.step()

    return all_losses, val_losses


def evaluate(model, data_loader, criterion=None, epoch=None):
    textlogger = logging.getLogger("openchem.evaluate")
    model.eval()
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
        logdir = model.module.logdir
    else:
        task = model.task
        eval_metrics = model.eval_metrics
        logdir = model.logdir

    for i_batch, sample_batched in enumerate(data_loader):
        if has_module:
            task = model.module.task
            use_cuda = model.module.use_cuda
            batch_input, batch_target = model.module.cast_inputs(sample_batched,
                                                                 task,
                                                                 use_cuda)
        else:
            task = model.task
            use_cuda = model.use_cuda
            batch_input, batch_target = model.cast_inputs(sample_batched, task, use_cuda)
        predicted = model(batch_input, eval=True)
        try:
            loss = criterion(predicted, batch_target)
        except TypeError:
            loss = 0.0
        if hasattr(predicted, 'detach'):
            predicted = predicted.detach().cpu().numpy()
        if hasattr(batch_target, 'cpu'):
            batch_target = batch_target.cpu().numpy()
        if hasattr(loss, 'item'):
            loss = loss.item()
        if isinstance(loss, list):
            loss = 0.0
        prediction += list(predicted)
        ground_truth += list(batch_target)
        loss_total += loss
        n_batches += 1
    
    cur_loss = loss_total / n_batches
    if task == 'classification':
        prediction = np.argmax(prediction, axis=1)
    if task == "graph_generation":
        f = open(logdir + "debug_smiles_epoch_" + str(epoch) + ".smi", "w")
        if isinstance(metrics, list) and len(metrics) == len(prediction):
            for i in range(len(prediction)):
                f.writelines(str(prediction[i]) + "," + str(metrics[i]) + "\n")
        else:
            for i in range(len(prediction)):
                f.writelines(str(prediction[i]) + "\n")
            f.close()
            
    metrics = calculate_metrics(prediction, ground_truth, eval_metrics)
    metrics = np.mean(metrics)

    if comm.is_main_process():
        textlogger.info('EVALUATION: [Time: %s, Loss: %.4f, Metrics: %.4f]' % (time_since(start), cur_loss, metrics))

    return cur_loss, metrics


def predict(model, data_loader, eval=True):
    textlogger = logging.getLogger("openchem.predict")
    model.eval()
    start = time.time()
    prediction = []
    samples = []
    has_module = False
    if hasattr(model, 'module'):
        has_module = True
    if has_module:
        task = model.module.task
        logdir = model.module.logdir
    else:
        task = model.task
        logdir = model.logdir

    for i_batch, sample_batched in enumerate(data_loader):
        if has_module:
            task = model.module.task
            use_cuda = model.module.use_cuda
            batch_input, batch_object = model.module.cast_inputs(sample_batched,
                                                                 task,
                                                                 use_cuda,
                                                                 for_prediction=True)
        else:
            task = model.task
            use_cuda = model.use_cuda
            batch_input, batch_object = model.cast_inputs(sample_batched,
                                                          task,
                                                          use_cuda,
                                                          for_predction=True)
        predicted = model(batch_input, eval=True)
        if hasattr(predicted, 'detach'):
            predicted = predicted.detach().cpu().numpy()
        prediction += list(predicted)
        samples += list(batch_object)

    if task == 'classification':
        prediction = np.argmax(prediction, axis=1)
    f = open(logdir + "/predictions.txt", "w")
    assert len(prediction) == len(samples)

    if comm.is_main_process():
        for i in range(len(prediction)):
            tmp = [chr(c) for c in samples[i]]
            tmp = ''.join(tmp)
            if " " in tmp:
                tmp = tmp[:tmp.index(" ")]
                to_write = [str(pred) for pred in prediction[i]]
                to_write = ",".join(to_write)
            f.writelines(tmp + "," + to_write + "\n")
        f.close()

    if comm.is_main_process():
        textlogger.info('Predictions saved to ' + logdir + "/predictions.txt")
        textlogger.info(
            'PREDICTION: [Time: %s, Number of samples: %d]' % (time_since(start), len(prediction))
        )


def reduce_tensor(tensor, world_size):
    r"""
    Reduces input ''tensor'' across all processes in such a way that everyone
    gets the sum of ''tensor'' from all of the processes.
    Args:
        tensor (Tensor): data to be reduced.
        world_size (int): number of processes.
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

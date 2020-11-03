# Modified from
# github.com/pytorch/fairseq/blob/master/fairseq/optim/fairseq_optimizer.py

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch.optim


class OpenChemOptimizer(object):
    def __init__(self, params, model_params):
        self.params = params[1]
        self._optimizer = params[0](model_params, **self.params)

    @property
    def optimizer(self):
        if not isinstance(self._optimizer, torch.optim.Optimizer):
            raise ValueError('_optimizer must be an instance of ' 'torch.optim.Optimizer')
        return self._optimizer

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def get_lr(self):
        """Return the current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def set_lr(self, lr):
        """Set the learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        """Return the optimizer's state dict."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Load an optimizer state dict.
        In general we should prefer the configuration of the existing optimizer
        instance (e.g., learning rate) over that found in the state_dict. This
        allows us to resume training from a checkpoint using a new set of
        optimizer args.
        """
        self.optimizer.load_state_dict(state_dict)

        # override learning rate, momentum, etc. with latest values
        for group in self.optimizer.param_groups:
            group.update(self.params)

    def step(self, closure=None):
        """Performs a single optimization step."""
        return self.optimizer.step(closure)

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        return self.optimizer.zero_grad()

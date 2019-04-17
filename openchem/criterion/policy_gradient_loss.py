import torch
from torch.nn import functional as F
from torch.nn import _Loss

import numpy as np

class PolicyGradientLoss(_Loss):
    def __init__(self, reward_fn, gamma=1.0):
        super(PolicyGradientLoss, self).__init__()
        self.reward_fn = reward_fn
        self.gamma = gamma

    def forward(self, input, target):
        device = input.device

        log_policy = input
        trajectories = target
        assert log_policy.size(0) == len(trajectories)
        len_trajectory = log_policy.size(1)
        batch_size = log_policy.size(0)
        with torch.no_grad():
            rewards = self.reward_fn(trajectories)
        rewards = np.asarray(rewards).reshape((batch_size, 1))
        discounts = np.power(gamma, np.arange(len_trajectory))
        discounts = discounts.reshape((len_trajectory, 1))
        discounted_rewards = np.outer(rewards, discounts)

        discounted_rewards = torch.tensor(discounted_rewards,
                                          dtype=torch.float,
                                          device=device)
        loss = discounted_rewards*log_policy
        loss = loss.sum(dim=1)
        loss = loss.mean(dim=0)
        return loss

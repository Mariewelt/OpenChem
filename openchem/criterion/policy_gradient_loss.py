import torch
from torch.nn.modules.loss import _Loss

from torch.nn.utils.rnn import pack_padded_sequence


class PolicyGradientLoss(_Loss):
    def __init__(self, reward_fn, critic, tokens, fn, gamma=1.0):
        super(PolicyGradientLoss, self).__init__()
        self.reward_fn = reward_fn
        self.gamma = gamma
        self.critic = critic
        self.tokens = tokens
        self.fn = fn

    def forward(self, input, target=None):

        log_policy = input[0]
        sizes = input[1]
        trajectories = input[2]

        device = log_policy.device

        len_trajectory = max(sizes)
        batch_size = len(sizes)
        with torch.no_grad():
            rewards = self.reward_fn(trajectories, self.critic,
                                     self.tokens, device, self.fn)
        rewards = rewards.view(batch_size, 1)
        discounts = torch.pow(
            self.gamma,
            torch.arange(len_trajectory, device=device, dtype=torch.float)
        )
        discounts = discounts.view(1, len_trajectory)
        discounted_rewards = rewards * discounts

        discounted_rewards = pack_padded_sequence(discounted_rewards,
                                                  sizes,
                                                  batch_first=True).data

        loss = - discounted_rewards * log_policy
        loss = loss.mean()
        return loss

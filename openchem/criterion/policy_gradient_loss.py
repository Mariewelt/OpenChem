import bisect
import numpy as np
import torch
from torch.nn.modules.loss import _Loss

from torch.nn.utils.rnn import pack_padded_sequence


class PolicyGradientLoss(_Loss):
    def __init__(self, reward_fn, critic, tokens, fn, gamma=1.0,
                 max_atom_bonds=None):
        super(PolicyGradientLoss, self).__init__()
        self.reward_fn = reward_fn
        self.gamma = gamma
        self.critic = critic
        self.tokens = tokens
        self.fn = fn
        self.max_atom_bonds = max_atom_bonds

    def forward(self, input, target=None):

        log_policy = input["log_policy"]
        sizes = input["sizes"]
        trajectories = input["smiles"]
        adj = input["adj"]
        classes = input["classes"]

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
        """
        if self.max_atom_bonds:
            atom_bonds = torch.cat(
                [torch.from_numpy(a).sum(dim=0) for a in adj]
            )
            classes = [classes[i // 2] if i % 2 == 1
                       else torch.tensor([0], dtype=torch.long)
                       for i in range(2 * len(classes))]
            classes = torch.cat(classes)
            max_atom_bonds = torch.tensor(self.max_atom_bonds)
            max_atom_bonds = max_atom_bonds[classes]
            structure_reward = (atom_bonds <= max_atom_bonds)

            # indices = (atom_bonds > max_atom_bonds).nonzero()
            # cum_sizes = np.cumsum(sizes)
            # sm_indices = [bisect.bisect_right(cum_sizes, i) for i in indices]
            # for i in np.unique(sm_indices):
            #     print(trajectories[i])

            discounted_rewards *= structure_reward
        """

        loss = - discounted_rewards * log_policy
        # print(loss.max().item())
        loss = loss.mean()
        return loss

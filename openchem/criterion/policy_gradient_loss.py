import bisect
import numpy as np
import torch
from torch.nn.modules.loss import _Loss

from torch.nn.utils.rnn import pack_padded_sequence


class PolicyGradientLoss(_Loss):
    def __init__(self, reward_fn, critic, tokens, fn, gamma=1.0,
                 max_atom_bonds=None, enable_supervised_loss=False):
        super(PolicyGradientLoss, self).__init__()
        self.reward_fn = reward_fn
        self.gamma = gamma
        self.critic = critic
        self.tokens = tokens
        self.fn = fn
        self.max_atom_bonds = max_atom_bonds
        self.enable_supervised_loss = enable_supervised_loss

    def forward(self, input, target=None):

        log_policy = input["log_policy"]
        sizes = input["sizes"]
        trajectories = input["smiles"]
        adj = input["adj"]
        classes = input["classes"]

        device = log_policy.device

        if self.critic is not None:
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
        # """
        if self.max_atom_bonds is not None:
            atom_bonds = torch.cat(
                [torch.from_numpy(a).sum(dim=0) for a in adj]
            )
            classes = [classes[i // 2] if i % 2 == 1
                       else torch.tensor([0], dtype=torch.long)
                       for i in range(2 * len(classes))]
            classes = torch.cat(classes)
            max_atom_bonds = torch.tensor(self.max_atom_bonds)
            max_atom_bonds = max_atom_bonds[classes]
            # objective: (atom_bonds <= max_atom_bonds)
            # structure_reward = -1 * (atom_bonds > max_atom_bonds).to(
            #     dtype=torch.float, device=device)
            structure_reward = (atom_bonds <= max_atom_bonds).to(
                dtype=torch.float, device=device)

            # indices = (atom_bonds > max_atom_bonds).nonzero()
            # cum_sizes = np.cumsum(sizes)
            # sm_indices = [bisect.bisect_right(cum_sizes, i) for i in indices]
            # for i in np.unique(sm_indices):
            #     print(trajectories[i])

            if self.critic is not None:
                discounted_rewards *= structure_reward
            else:
                discounted_rewards = structure_reward
        # """

        loss = - discounted_rewards * log_policy
        # print(loss.max().item())
        loss = loss.mean()
        if self.enable_supervised_loss:
            loss = loss + input["loss"]
        return loss

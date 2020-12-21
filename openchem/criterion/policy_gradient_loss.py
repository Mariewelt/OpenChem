import bisect
import bisect
import numpy as np
import torch
from torch.nn.modules.loss import _Loss

from torch.nn.utils.rnn import pack_padded_sequence

from openchem.data.utils import sanitize_smiles


class PolicyGradientLoss(_Loss):
    def __init__(self, reward_fn, critic, fn, tokens=None, gamma=1.0, max_atom_bonds=None, enable_supervised_loss=False):
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
        len_trajectory = max(sizes)
        batch_size = len(sizes)

        if self.critic is not None:
            # Current convention is to run critic only on valid molecules
            # Others receive zero reward from the critic

            clean_smiles, clean_idx = sanitize_smiles(trajectories, allowed_tokens=self.tokens, logging="none")
            clean_smiles = [clean_smiles[i] for i in clean_idx]

            with torch.no_grad():
                clean_rewards = self.reward_fn(clean_smiles, self.critic, self.tokens, device, self.fn)

            rewards = torch.zeros((batch_size, *clean_rewards.shape[1:]),
                                  dtype=clean_rewards.dtype,
                                  device=clean_rewards.device)
            clean_idx = torch.tensor(clean_idx, device=clean_rewards.device)
            rewards.index_copy_(0, clean_idx, clean_rewards)

            rewards = rewards.view(batch_size, 1)
            discounts = torch.pow(self.gamma, torch.arange(len_trajectory, device=device, dtype=torch.float))
            discounts = discounts.view(1, len_trajectory)
            discounted_rewards = rewards * discounts

            discounted_rewards = pack_padded_sequence(discounted_rewards, sizes, batch_first=True).data

        sanitize_smiles(trajectories, allowed_tokens=self.tokens, logging="info")

        if self.max_atom_bonds is not None:

            structure_reward = torch.zeros((batch_size, len_trajectory),
                                           dtype=log_policy.dtype,
                                           device=log_policy.device)
            for i in range(batch_size):
                atom_bonds = torch.from_numpy(adj[i]).sum(dim=0)
                cl = torch.cat([torch.tensor([0], dtype=torch.long), classes[i]])
                max_atom_bonds = torch.tensor(self.max_atom_bonds)
                max_atom_bonds = max_atom_bonds[cl]

                # structure_reward[i, :sizes[i]] = \
                #     (atom_bonds <= max_atom_bonds).to(
                #         dtype=torch.float, device=device)
                structure_reward[i, :sizes[i]] = \
                    -15. * (atom_bonds > max_atom_bonds).to(
                        dtype=torch.float, device=device)

            structure_reward = pack_padded_sequence(structure_reward, sizes, batch_first=True).data

            if self.critic is not None:
                discounted_rewards += structure_reward
            else:
                discounted_rewards = structure_reward

        loss = -discounted_rewards * log_policy
        loss = loss.mean()
        if self.enable_supervised_loss:
            loss = loss + input["loss"]
        return loss

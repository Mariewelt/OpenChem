import torch
import numpy as np
from openchem.data.utils import process_smiles


def melt_t_max_fn(prediction):
    rewards = torch.exp(prediction + 1.0)
    return rewards


def reward_fn(smiles, predictor, old_tokens, device, fn):

    clean_smiles, _, length, tokens, _, _, clean_idx = process_smiles(
        smiles,
        target=None,
        augment=False,
        tokens=old_tokens,
        pad=True,
        tokenize=True,
        flip=False,
        return_idx=True,
        allowed_tokens=old_tokens
    )
    num_all = len(clean_smiles)
    clean_smiles = clean_smiles[np.array(clean_idx)]
    length = [length[i] for i in clean_idx]

    smiles_tensor = torch.from_numpy(clean_smiles).to(
        dtype=torch.long, device=device)
    length_tensor = torch.tensor(length, dtype=torch.long, device=device)
    prediction = predictor([smiles_tensor, length_tensor], eval=True)
    if predictor.task == "classification":
        prediction = torch.argmax(prediction, dim=1)

    rewards = fn(prediction.to(torch.float))
    rewards_all = torch.zeros((num_all, *rewards.shape[1:]),
                              dtype=rewards.dtype, device=rewards.device)
    clean_idx = torch.tensor(clean_idx, device=rewards.device)
    rewards_all.index_copy_(0, clean_idx, rewards)

    return rewards

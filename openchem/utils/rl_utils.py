import torch
import numpy as np
from openchem.data.utils import process_smiles


def melt_t_max_fn(prediction):
    rewards = torch.exp(prediction + 1.0)
    return rewards


def reward_fn(smiles, predictor, old_tokens, device, fn):

    clean_smiles, _, length, tokens, _, _ = process_smiles(
        smiles,
        sanitized=True,
        target=None,
        augment=False,
        tokens=old_tokens,
        pad=True,
        tokenize=True,
        flip=False,
        allowed_tokens=old_tokens
    )

    smiles_tensor = torch.from_numpy(clean_smiles).to(
        dtype=torch.long, device=device)
    length_tensor = torch.tensor(length, dtype=torch.long, device=device)
    prediction = predictor([smiles_tensor, length_tensor], eval=True)
    if predictor.task == "classification":
        prediction = torch.argmax(prediction, dim=1)

    rewards = fn(prediction.to(torch.float))

    return rewards


def qed_max_rew(prediction):
    return prediction*10.0


def qed_reward_fn(smiles, critic, old_tokens, device, fn):

    prediction = critic(smiles, return_mean=False)
    prediction = torch.from_numpy(np.array(prediction)).to(
        dtype=torch.float, device=device)

    rewards = fn(prediction)

    return rewards

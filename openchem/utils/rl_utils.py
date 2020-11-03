import torch
import numpy as np
from openchem.data.utils import process_smiles, process_graphs
from openchem.utils.graph import Attribute


def melt_t_max_fn(prediction):
    #positive = torch.torch.full_like(prediction, 5.)
    #negative = torch.torch.full_like(prediction, -5.)
    #rewards = torch.where(prediction > 3., positive, negative)
    #rewards = prediction*5
    #rewards = torch.where(rewards > 10., positive, rewards)
    #rewards = torch.where(rewards < -10., negative, rewards)
    rewards = torch.exp(prediction + 1.0)
    return rewards


def get_atomic_attributes(atom):
    attr_dict = {}

    atomic_num = atom.GetAtomicNum()
    atomic_mapping = {5: 0, 7: 1, 6: 2, 8: 3, 9: 4, 15: 5, 16: 6, 17: 7, 35: 8, 53: 9}
    if atomic_num in atomic_mapping.keys():
        attr_dict['atom_element'] = atomic_mapping[atomic_num]
    else:
        attr_dict['atom_element'] = 10
    attr_dict['valence'] = atom.GetTotalValence()
    attr_dict['charge'] = atom.GetFormalCharge()
    attr_dict['hybridization'] = atom.GetHybridization().real
    attr_dict['aromatic'] = int(atom.GetIsAromatic())
    return attr_dict


def graph_reward_fn(smiles, predictor, tokens, device, fn):
    node_attributes = {}
    node_attributes['valence'] = Attribute('node', 'valence', one_hot=True, values=[1, 2, 3, 4, 5, 6, 7])

    node_attributes['charge'] = Attribute('node', 'charge', one_hot=True, values=[-1, 0, 1, 2, 3, 4])

    node_attributes['hybridization'] = Attribute('node',
                                                 'hybridization',
                                                 one_hot=True,
                                                 values=[0, 1, 2, 3, 4, 5, 6, 7])

    node_attributes['aromatic'] = Attribute('node', 'aromatic', one_hot=True, values=[0, 1])

    node_attributes['atom_element'] = Attribute('node', 'atom_element', one_hot=True, values=list(range(11)))

    adj_matrix, node_feature_matrix = process_graphs(smiles,
                                                     node_attributes,
                                                     get_atomic_attributes,
                                                     edge_attributes=None,
                                                     get_bond_attributes=None)
    adj_matrix = np.array(adj_matrix)
    node_feature_matrix = np.array(node_feature_matrix)
    adj_tensor = torch.from_numpy(adj_matrix).to(dtype=torch.float32, device=device)
    node_feature_tensor = torch.from_numpy(node_feature_matrix).to(dtype=torch.float32, device=device)
    inp = (node_feature_tensor, adj_tensor)
    prediction = predictor(inp, eval=True)

    if predictor.task == "classification":
        prediction = torch.argmax(prediction, dim=1)

    rewards = fn(prediction.to(torch.float))

    return rewards


def reward_fn(smiles, predictor, old_tokens, device, fn, eval=False):

    if eval:
        sanitized = False
    else:
        sanitized = True
    if old_tokens is None:
        sanitized = True
    clean_smiles, _, length, tokens, _, _ = process_smiles(smiles,
                                                           sanitized=sanitized,
                                                           target=None,
                                                           augment=False,
                                                           tokens=old_tokens,
                                                           pad=True,
                                                           tokenize=True,
                                                           flip=False,
                                                           allowed_tokens=old_tokens)
    smiles_tensor = torch.from_numpy(clean_smiles).to(dtype=torch.long, device=device)
    length_tensor = torch.tensor(length, dtype=torch.long, device=device)
    prediction = predictor([smiles_tensor, length_tensor], eval=True)
    if predictor.task == "classification":
        prediction = torch.argmax(prediction, dim=1)

    rewards = fn(prediction.to(torch.float))

    return rewards


def qed_max_rew(prediction):
    return prediction * 10.0


def logp_pen_rew(prediction):
    return prediction * 5.0


def logp_range_rew(prediction):
    positive = torch.ones_like(prediction)
    negative = torch.full_like(prediction, -1.)
    rewards = torch.where((prediction >= 0.) & (prediction <= 5.), positive, negative)
    return rewards


def rocs_reward_fn(smiles, critic, old_tokens, device, fn):
    
    prediction = critic(smiles)
    prediction = torch.from_numpy(np.array(prediction)).to(dtype=torch.float, device=device)
   
    rewards = fn(prediction)
    return rewards


def qed_reward_fn(smiles, critic, old_tokens, device, fn):

    prediction = critic(smiles, return_mean=False)
    prediction = torch.from_numpy(np.array(prediction)).to(dtype=torch.float, device=device)

    rewards = fn(prediction)

    return rewards


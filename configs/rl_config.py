import torch
import pickle
from openchem.utils.rl_utils import reward_fn, melt_t_max_fn
from openchem.models.Smiles2Label import Smiles2Label

params = pickle.load(open('./logs/melt_temp_rnn_log/params.pkl', 'rb'))

model = Smiles2Label(params).cuda()
weights = torch.load('./logs/melt_temp_rnn_log/checkpoint/epoch_30')
new_weights = {}

for key in weights.keys():
    new_weights[key[7:]] = weights[key]

model.load_state_dict(new_weights)

tokens = params['tokens']

from openchem.criterion.policy_gradient_loss import PolicyGradientLoss
my_loss = PolicyGradientLoss(reward_fn=reward_fn, critic=model, tokens=tokens, fn=melt_t_max_fn, gamma=0.97)

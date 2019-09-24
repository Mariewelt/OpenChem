import time
import math
from six import string_types
from collections import defaultdict

import torch
import glob
import os
import six


def move_to_cuda(sample):
    # copy-pasted from
    # https://github.com/pytorch/fairseq/blob/master/fairseq/utils.py
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda()
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_cuda(sample)


def get_latest_checkpoint(path):
    if os.path.isdir(path) and os.listdir(path) != []:
        list_of_files = glob.glob(os.path.join(path, '*'))
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file
    else:
        return None


def deco_print(line, offset=0, start="*** ", end='\n'):
    # copy-pasted from
    # github.com/NVIDIA/OpenSeq2Seq/blob/master/open_seq2seq/utils/utils.py
    if six.PY2:
        print((start + " " * offset + line).encode('utf-8'), end=end)
    else:
        print(start + " " * offset + line, end=end)


def flatten_dict(dct):
    # copy-pasted from
    # github.com/NVIDIA/OpenSeq2Seq/blob/master/open_seq2seq/utils/utils.py
    flat_dict = {}
    for key, value in dct.items():
        if isinstance(value, int) or isinstance(value, float) or \
           isinstance(value, string_types) or isinstance(value, bool):
            flat_dict.update({key: value})
        elif isinstance(value, dict):
            flat_dict.update({key + '/' + k: v for k, v in flatten_dict(dct[key]).items()})
    return flat_dict


def nest_dict(flat_dict):
    # copy-pasted from
    # github.com/NVIDIA/OpenSeq2Seq/blob/master/open_seq2seq/utils/utils.py
    nst_dict = {}
    for key, value in flat_dict.items():
        nest_keys = key.split('/')
        cur_dict = nst_dict
        for i in range(len(nest_keys) - 1):
            if nest_keys[i] not in cur_dict:
                cur_dict[nest_keys[i]] = {}
            cur_dict = cur_dict[nest_keys[i]]
        cur_dict[nest_keys[-1]] = value
    return nst_dict


def nested_update(org_dict, upd_dict):
    # copy-pasted from
    # github.com/NVIDIA/OpenSeq2Seq/blob/master/open_seq2seq/utils/utils.py
    for key, value in upd_dict.items():
        if isinstance(value, dict):
            if key in org_dict:
                if not isinstance(org_dict[key], dict):
                    raise ValueError("Mismatch between org_dict and upd_dict " "at node {}".format(key))
                nested_update(org_dict[key], value)
            else:
                org_dict[key] = value
        else:
            org_dict[key] = value


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60

    return '%dm %ds' % (m, s)


def identity(input):
    return input


def calculate_metrics(predicted, ground_truth, metrics):
    return metrics(ground_truth, predicted)


def check_params(config, required_dict, optional_dict):
    if required_dict is None or optional_dict is None:
        return

    for pm, vals in required_dict.items():
        if pm not in config:
            raise ValueError("{} parameter has to be specified".format(pm))
        else:
            if vals == str:
                vals = string_types
            if vals and isinstance(vals, list) and config[pm] not in vals:
                raise ValueError("{} has to be one of {}".format(pm, vals))
            if vals and not isinstance(vals, list) and \
                    not isinstance(config[pm], vals):
                raise ValueError("{} has to be of type {}".format(pm, vals))

    for pm, vals in optional_dict.items():
        if vals == str:
            vals = string_types
        if pm in config:
            if vals and isinstance(vals, list) and config[pm] not in vals:
                raise ValueError("{} has to be one of {}".format(pm, vals))
            if vals and not isinstance(vals, list) and \
                    not isinstance(config[pm], vals):
                raise ValueError("{} has to be of type {}".format(pm, vals))

    # for pm in config:
    #     if pm not in required_dict and pm not in optional_dict:
    #         raise ValueError("Unknown parameter: {}".format(pm))


def cross_validation_split(data, targets, n_folds=5, split='random', stratified=True, folds=None):
    if split not in ['random', 'fixed']:
        raise ValueError('Invalid value for argument \'split\': ' 'must be either \'random\' or \'fixed\'.')
    if split == 'fixed' and folds is None:
        raise ValueError('When \'split\' is \'fixed\' ' 'argument \'folds\' must be provided.')
    if split == 'fixed':
        assert len(targets) == len(folds)
    raise NotImplementedError


def make_positions(tensor, padding_idx, left_pad):
    # Copy-pasted from
    # https://github.com/pytorch/fairseq/blob/master/fairseq/utils.py
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """
    max_pos = padding_idx + 1 + tensor.size(1)
    if not hasattr(make_positions, 'range_buf'):
        make_positions.range_buf = tensor.new()
    make_positions.range_buf = make_positions.range_buf.type_as(tensor)
    if make_positions.range_buf.numel() < max_pos:
        torch.arange(padding_idx + 1, max_pos, out=make_positions.range_buf)
    mask = tensor.ne(padding_idx)
    positions = make_positions.range_buf[:tensor.size(1)].expand_as(tensor)
    if left_pad:
        positions = positions - mask.size(1) + \
                    mask.long().sum(dim=1).unsqueeze(1)
    return tensor.clone().masked_scatter_(mask, positions[mask])

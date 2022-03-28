import logging
import math
from typing import Dict

import yaml

import torch


def load_config(path):
    """
    Loads config file:

    Args:
        path (str): path to the config file

    Returns:
        config (dict): dictionary of the configuration parameters, merge sub_dicts

    """
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    config = dict()
    for key, value in cfg.items():
        for k, v in value.items():
            config[k] = v

    return config


def metrics_to_string(metrics, prefix=None):
    """Can also be used for losses"""
    s = ', '.join([f"{k}: {metrics[k]:.4g}" for k in sorted(list(metrics)) if metrics[k].ndim == 0])
    if prefix is not None:
        s = prefix + ' ' + s
    return s


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name=None):
        self.reset()
        self.name = name
        self.logger = logging.getLogger(__name__)

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0.0
        self.sq_sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()

        if math.isnan(val):
            self.logger.warning(f'Trying to update Average Meter {self.name} with invalid value, ignoring...')
            return

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.sq_sum += val ** 2 * n
        self.var = self.sq_sum / self.count - self.avg ** 2

    def __repr__(self):
        return 'N: {}, avg: {:.3g}'.format(self.count, self.avg)


class StatsMeter(object):
    """Dictionary of AverageMeters"""
    def __init__(self):
        self.meters: Dict[AverageMeter] = {}

    def __getitem__(self, item) -> AverageMeter:
        if item not in self.meters:
            self.meters[item] = AverageMeter(item)
        return self.meters[item]

    def __iter__(self):
        return iter(self.meters.keys())

    def clear(self):
        self.meters = {}

    def items(self):
        return self.meters.items()

    def __repr__(self):
        repr = 'StatsMeter containing the following fields:\n'
        fields = ['[{}] {} '.format(k, self.meters[k].__str__()) for k in self.meters]
        repr += ' | '.join(fields)
        return repr


def metrics_to_string(metrics, prefix=None):
    """Can also be used for losses"""
    s = ', '.join([f"{k}: {metrics[k]:.4g}" for k in sorted(list(metrics)) if
                   isinstance(metrics[k], float) or metrics[k].ndim == 0])
    if prefix is not None:
        s = prefix + ' ' + s
    return s


def stack_lengths_to_batch_indices(stack_lengths, device=None):
    """
    Example:
        [2, 3] -> [0, 0, 1, 1, 1]"""
    return torch.cat([torch.tensor([b] * l, dtype=torch.int64, device=device)
                      for (b, l) in enumerate(stack_lengths)])
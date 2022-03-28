import inspect
import logging
import os
import importlib

import torch

_MODELS = {}


def add_models(fname):
    clsmembers = inspect.getmembers(importlib.import_module(f'models.{fname}'),
                                    inspect.isclass)
    _MODELS.update({f'{fname}.{m[0]}':m[1] for m in clsmembers if issubclass(m[1], torch.nn.Module)})


# Loads all modules in this file dynamically
files_in_folder = list(filter(lambda x: x.endswith('.py') and not x.startswith('_'),
                              os.listdir(os.path.dirname(__file__))))
for fname in files_in_folder:
    add_models(os.path.splitext(fname)[0])


def get_model(model_name):
    if model_name not in _MODELS:
        logging.error('Model not found, options are {}'.format(_MODELS.keys()))
        return None

    Model = _MODELS[model_name]
    return Model
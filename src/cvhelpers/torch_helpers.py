"""PyTorch related utility functions
"""

import logging
import importlib
import os
import pdb
import random
import shutil
import sys
import time
import traceback
from typing import Union, List

import numpy as np
import torch

# optional imports for less common libraries
try:
    import torch_geometric
    _torch_geometric_exists = True
except ImportError:
    _torch_geometric_exists = False


def all_to_device(data, device):
    """Sends everything into a certain device """
    if isinstance(data, dict):
        for k in data:
            data[k] = all_to_device(data[k], device)
        return data
    elif isinstance(data, list):
        data = [all_to_device(d, device) for d in data]
        return data
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    elif _torch_geometric_exists and isinstance(data, torch_geometric.data.batch.Batch):
        return data.to(device)
    else:
        return data  # Cannot be converted


def to_numpy(tensor: Union[np.ndarray, torch.Tensor, List]) -> Union[np.ndarray, List]:
    """Wrapper around .detach().cpu().numpy() """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, list):
        return [to_numpy(l) for l in tensor]
    elif isinstance(tensor, str):
        return tensor
    elif tensor is None:
        return None
    else:
        raise NotImplementedError


def all_isfinite(x):
    """Check the entire nested dictionary/list of tensors is finite
    (i.e. not nan or inf)"""
    if isinstance(x, torch.Tensor):
        return bool(torch.all(torch.isfinite(x)))
    elif isinstance(x, list):
        return all([all_isfinite(xi) for xi in x])
    elif isinstance(x, list):
        return all([all_isfinite(x[k]) for k in x])

    # If it reaches here, it's an unsupported type. Returns True for such cases
    return True


def seed_numpy_fn(x):
    """Numpy random seeding function to pass into Pytorch's dataloader.

    This is required since numpy RNG is incompatible with fork
    https://pytorch.org/docs/stable/notes/faq.html#my-data-loader-workers-return-identical-random-numbers

    Example usage:
        DataLoader(..., worker_init_fn=seed_numpy_fn)
    """
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)


def setup_seed(seed, cudnn_deterministic=False):
    """
    fix random seed for deterministic training
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True


class CheckPointManager(object):
    """Manager for saving/managing pytorch checkpoints.

    Provides functionality similar to tf.Saver such as
    max_to_keep and keep_checkpoint_every_n_hours
    """
    def __init__(self, save_path: str = None, max_to_keep=3, keep_checkpoint_every_n_hours=6.0):

        if max_to_keep <= 0:
            raise ValueError('max_to_keep must be at least 1')

        self._max_to_keep = max_to_keep
        self._keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours

        self._logger = logging.getLogger(self.__class__.__name__)
        self._checkpoints_permanent = []  # Will not be deleted
        self._checkpoints_buffer = []  # Those which might still be deleted
        self._next_save_time = time.time()
        self._best_score = None
        self._best_step = None

        if save_path is not None:
            self._ckpt_dir = os.path.dirname(save_path)
            self._save_path = save_path + '-{}.pth'
            self._checkpoints_fname = os.path.join(self._ckpt_dir, 'checkpoints.txt')
            os.makedirs(self._ckpt_dir, exist_ok=True)
            self._update_checkpoints_file()
        else:
            self._ckpt_dir = None
            self._save_path = None
            self._checkpoints_fname = None

    def _save_checkpoint(self, step, model, score, **kwargs):
        save_name = self._save_path.format(step)

        model_state_dict = {k: v for (k, v) in model.state_dict().items() if not v.is_sparse}
        state = {'state_dict': model_state_dict,
                 'step': step}
        for k in kwargs:
            if getattr(kwargs[k], 'state_dict', None) is not None:
                state[k] = kwargs[k].state_dict()
            else:
                state[k] = kwargs[k]  # Note that loading of this variable is not supported

        torch.save(state, save_name)
        self._logger.info('Saved checkpoint: {}'.format(save_name))

        self._checkpoints_buffer.append((save_name, time.time(), step))

        if self._best_score is None or np.all(np.array(score) >= np.array(self._best_score)):
            # Remove previous best checkpoint if no longer best
            if self._best_score is not None and \
                    self._best_step not in [c[2] for c in self._checkpoints_buffer] and \
                    self._best_step not in [c[2] for c in self._checkpoints_permanent]:
                os.remove(self._save_path.format(self._best_step))

            self._best_score = score
            self._best_step = step
            self._logger.info('Checkpoint is current best, score={}'.format(
                np.array_str(np.array(self._best_score), precision=3)))

    def _remove_old_checkpoints(self):
        while len(self._checkpoints_buffer) > self._max_to_keep:
            to_remove = self._checkpoints_buffer.pop(0)

            if to_remove[1] > self._next_save_time:
                self._checkpoints_permanent.append(to_remove)
                self._next_save_time = to_remove[1] + self._keep_checkpoint_every_n_hours * 3600
            else:
                # Remove old checkpoint unless it's the best
                if self._best_step != to_remove[2]:
                    os.remove(to_remove[0])

    def _update_checkpoints_file(self):
        checkpoints = [os.path.basename(c[0]) for c in self._checkpoints_permanent + self._checkpoints_buffer]
        with open(self._checkpoints_fname, 'w') as fid:
            fid.write('Best step: {}'.format(self._best_step) + '\n')
            fid.write('\n'.join(checkpoints))


    def save(self, model: torch.nn.Module, step: int, score: float = 0.0,
             **kwargs):
        """Save model checkpoint to file

        Args:
            model: Torch model
            step (int): Step, model will be saved as model-[step].pth
            score (float, optional): To determine which model is the best (i.e. highest score)
            **kwargs: For saving arbitrary data, e.g. for optimizer or scheduler.
        """
        if self._save_path is None:
            raise AssertionError('Checkpoint manager must be initialized with save path for save().')

        self._save_checkpoint(step, model, score, **kwargs)
        self._remove_old_checkpoints()
        self._update_checkpoints_file()

    def load(self, save_path, model: torch.nn.Module = None, 
             **kwargs):
        """Loads saved model from file

        Args:
            save_path: Path to saved model (.pth). If a directory is provided instead, the
              best checkpoint is used instead.
            model: Torch model to restore weights to
            **kwargs: For loading arbitrary data, e.g. for optimizer or scheduler. Inputs must have
              load_state_dict() method.
        """
        if os.path.isdir(save_path):
            # Get best step, which is stored in the first line of the checkpoints file
            with open(os.path.join(save_path, 'checkpoints.txt')) as fid:
                line = fid.readline()
                assert line.startswith('Best'), 'checkpoints.txt not in expected format.'
                best_step = int(line.split(':')[1])
                save_path = os.path.join(save_path, 'model-{}.pth'.format(best_step))

        if not torch.cuda.is_available():
            state = torch.load(save_path, map_location=torch.device('cpu'))
        else:
            state = torch.load(save_path)

        step = state.get('step', 0)

        if 'state_dict' in state and model is not None:
            retval = model.load_state_dict(state['state_dict'], strict=False)
            if len(retval.unexpected_keys) > 0:
                self._logger.warning('Unexpected keys in checkpoint: {}'.format(
                    retval.unexpected_keys))
            if len(retval.missing_keys) > 0:
                self._logger.warning('Missing keys in checkpoint: {}'.format(
                    retval.missing_keys))

        for k in kwargs:
            try:
                if k in state and getattr(kwargs[k], 'load_state_dict', None) is not None:
                    kwargs[k].load_state_dict(state[k])
                else:
                    self._logger.warning(f'"{k}" ignored from checkpoint loading')
            except ValueError as e:
                self._logger.error(
                    'Loading {} from checkpoint failed due to error "{}", but ignoring and proceeding...'.format(k, e))


        self._logger.info('Loaded models from {}'.format(save_path))
        return step


class TorchDebugger(torch.autograd.detect_anomaly):
    """Enters debugger when anomaly detected"""
    def __enter__(self) -> None:
        super().__enter__()

    def __exit__(self, type, value, trace):
        super().__exit__()
        if isinstance(value, RuntimeError):
            traceback.print_tb(trace)
            print(value)
            if sys.gettrace() is None:
                pdb.set_trace()
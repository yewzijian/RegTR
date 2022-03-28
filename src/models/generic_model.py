"""Generic model
"""
import logging
from typing import Tuple

import torch.nn
from torch.utils.tensorboard import SummaryWriter


class GenericModel(torch.nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.device = None
        self.logger = logging.getLogger(self.__class__.__name__)

        self.optimizer = None
        self.scheduler = None
        self.optimizer_handled_by_trainer = True
        self._trainer = None
        self.logger.info(f'Instantiating model {self.__class__.__name__}')

    def set_trainer(self, trainer):
        self._trainer = trainer

    def get_trainer(self):
        """Returns the trainer instance"""
        return self._trainer

    def train_epoch_start(self):
        pass

    def training_step(self, batch, batch_idx):
        """Training step.

        Returns:
            losses(Dict): Which should be a python dictionary and should have at
              least one term 'total' for the total loss
        """
        raise NotImplementedError

    def train_epoch_end(self):
        pass

    def train_summary_fn(self, writer: SummaryWriter, step: int,
                         data_batch, train_output, train_losses):
        self._generic_summary_function(writer, step, model=self, losses=train_losses)

    def validation_epoch_start(self):
        pass

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_epoch_end(self, validation_step_outputs) -> Tuple[float, dict]:
        pass

    def validation_summary_fn(self, writer: SummaryWriter, step: int, val_outputs):
        """Logs data during validation. This function will be called after every
        validation run.
        The default implementation saves out the scalars from losses and metrics.

        Args:
            writer: validation writer
            step: The current step number
            val_outputs: Whatever that is returned from validation_epoch_end()

        """
        if isinstance(val_outputs, dict):
            self._generic_summary_function(writer, step, **val_outputs)

    def test_epoch_start(self):
        pass

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_epoch_end(self, test_step_outputs):
        pass

    def configure_optimizers(self):
        """Sets and returns the optimizers. Default implementation does nothing.
        """
        pass

    def to(self, *args, **kwargs):
        """Sends the model to the specified device. Also sets self.device
        so that it can be accessed by code within the model.
        """
        super().to(*args, **kwargs)

        # Keep track of device in an easily accessible place
        if 'device' in kwargs:
            self.device = kwargs['device']
        else:
            self.device = args[0]
        return self

    def _generic_summary_function(self, writer: SummaryWriter, step: int, **kwargs):
        # Generic summary function that saves losses and metrics as tensorboard
        # summary scalars.
        losses = kwargs.get('losses', None)
        if losses is not None:
            for k in losses:
                if isinstance(losses[k], torch.Tensor) and losses[k].ndim > 0:
                    continue
                writer.add_scalar('losses/{}'.format(k), losses[k], step)

        metrics = kwargs.get('metrics', None)
        if metrics is not None:
            for k in metrics:
                if isinstance(metrics[k], torch.Tensor) and metrics[k].ndim > 0:
                    continue
                writer.add_scalar('metrics/{}'.format(k), metrics[k], step)

        if self.scheduler is not None:
            writer.add_scalar('lr', self.scheduler.get_last_lr()[0], step)

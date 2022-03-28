import math


class WarmUpScheduler:

    def __init__(self, optimizer, params, max_lr):
        """Implements learning rate warm up for transformer post norm

        Args:
            optimizer:
            params: [warmup_steps, num_decay_steps, decay_factor]
            max_lr:
        """
        self.optimizer = optimizer

        self.warmup_steps = params[0]
        if len(params) == 1:
            self.gamma = 1.0
        else:
            self.gamma = math.exp(math.log(params[2]) / params[1])
        self.max_lr = max_lr
        self._step = 0
        self._lr = 0  # Current last learning rate

    def state_dict(self):
        """Returns the state of the warmup scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the warmup scheduler's state.
        Arguments:
            state_dict (dict): warmup scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def step(self):
        "Update parameters and rate"
        self._step += 1
        lr = self.compute_lr()
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        self._lr = lr
        self.optimizer.step()

    def compute_lr(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step

        if step < self.warmup_steps:  # Warmup phase
            return min(step / self.warmup_steps, 1.0) * self.max_lr
        else:  # decay phase
            return math.pow(self.gamma, step - self.warmup_steps) * self.max_lr

    def get_last_lr(self):
        return [self._lr]

    def __repr__(self):
        return f'WarmUpScheduler with (warmup_steps={self.warmup_steps}, max lr={self.max_lr})'
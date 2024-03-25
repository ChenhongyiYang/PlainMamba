import copy
from typing import Optional

import os.path as osp
from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks.hook import HOOKS, Hook
from mmcv.fileio import FileClient

from mmcv.runner.dist_utils import allreduce_params, master_only
from mmcv.runner.hooks.checkpoint import CheckpointHook
from mmcv.runner.checkpoint import save_checkpoint


@HOOKS.register_module()
class EMACheckpointHook(Hook):
    def __init__(
        self,
        # EMA
        momentum: float = 0.0002,
        interval: int = 1,
        warm_up: int = 100,
        resume_from: Optional[str] = None,
        # Checkpoint
        save_interval: int = -1,
        by_epoch: bool = True,
        save_optimizer: bool = True,
        out_dir: Optional[str] = None,
        max_keep_ckpts: int = -1,
        save_last: bool = True,
        sync_buffer: bool = False,
        file_client_args: Optional[dict] = None,
        decay_epochs=(250,),
        decay_factor=10.,
        **kwargs
    ):
        assert isinstance(interval, int) and interval > 0
        self.warm_up = warm_up
        self.interval = interval
        assert momentum > 0 and momentum < 1
        self.momentum = momentum**interval
        self.checkpoint = resume_from

        # Checkpoint
        self.save_interval = save_interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.max_keep_ckpts = max_keep_ckpts
        self.save_last = save_last
        self.args = kwargs
        self.sync_buffer = sync_buffer
        self.file_client_args = file_client_args

        self.decay_epochs = decay_epochs
        self.decay_factor = decay_factor

        assert by_epoch

    def before_run(self, runner):
        """To resume model with it's ema parameters more friendly.

        Register ema parameter as ``named_buffer`` to model
        """
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        self.param_ema_buffer = {}
        self.model_parameters = dict(model.named_parameters(recurse=True))
        for name, value in self.model_parameters.items():
            # "." is not allowed in module's buffer name
            buffer_name = f"ema_{name.replace('.', '_')}"
            self.param_ema_buffer[name] = buffer_name
            model.register_buffer(buffer_name, value.data.clone())
        self.model_buffers = dict(model.named_buffers(recurse=True))
        if self.checkpoint is not None:
            runner.resume(self.checkpoint)
        # Checkpoint
        if not self.out_dir:
            self.out_dir = runner.work_dir
        self.file_client = FileClient.infer_client(self.file_client_args, self.out_dir)
        if self.out_dir != runner.work_dir:
            basename = osp.basename(runner.work_dir.rstrip(osp.sep))
            self.out_dir = self.file_client.join_path(self.out_dir, basename)

    def after_train_iter(self, runner):
        """Update ema parameter every self.interval iterations."""
        curr_step = runner.iter
        # We warm up the momentum considering the instability at beginning
        momentum = min(self.momentum,
                       (1 + curr_step) / (self.warm_up + curr_step))
        momentum = self._decay_momentum(momentum, runner.epoch)
        if curr_step % self.interval != 0:
            return
        for name, parameter in self.model_parameters.items():
            buffer_name = self.param_ema_buffer[name]
            buffer_parameter = self.model_buffers[buffer_name]
            buffer_parameter.mul_(1 - momentum).add_(momentum, parameter.data)

    def _decay_momentum(self, momentum, curr_epoch):
        for i, _e in enumerate(self.decay_epochs):
            if curr_epoch < _e:
                return momentum * self.decay_factor**float(i)

    def after_train_epoch(self, runner):
        self._swap_ema_parameters()

    def before_train_epoch(self, runner):
        """We recover model's parameter from ema backup after last epoch's
        EvalHook."""
        self._swap_ema_parameters()

    def _swap_ema_parameters(self):
        """Swap the parameter of model with parameter in ema_buffer."""
        for name, value in self.model_parameters.items():
            temp = value.data.clone()
            ema_buffer = self.model_buffers[self.param_ema_buffer[name]]
            value.data.copy_(ema_buffer.data)
            ema_buffer.data.copy_(temp)

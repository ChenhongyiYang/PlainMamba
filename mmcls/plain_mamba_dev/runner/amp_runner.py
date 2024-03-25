'''
Author: Chenhongyi Yang
'''
import copy
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch._six import inf
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.cuda.amp.grad_scaler import OptState
from torch.cuda.amp import autocast

from mmcv.runner.epoch_based_runner import EpochBasedRunner
from mmcv.runner.iter_based_runner import IterBasedRunner
from mmcv.runner import get_dist_info
from mmcv.runner.builder import RUNNERS
from mmcv.runner.utils import get_host_info
from mmcv.runner.hooks.optimizer import Fp16OptimizerHook, GradientCumulativeFp16OptimizerHook

from mmcls.core.utils.dist_utils import DistOptimizerHook

from torch.optim import Adam



@RUNNERS.register_module()
class AmpEpochBasedRunner(EpochBasedRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler


    def _amp_train_step(self, data_batch, **kwargs):
        with autocast():
            self.run_iter(data_batch, train_mode=True, **kwargs)
            losses = self.outputs['loss']
        self.optimizer.zero_grad()
        self.grad_scaler.scale(losses).backward()
        if self.grad_clip is not None:
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip['max_norm'])
        else:
            raise NotImplementedError("AmpEpochBasedRunner should be used with grad_clip")

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self.data_batch = data_batch
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self._amp_train_step(data_batch, **kwargs)
            self.call_hook('after_train_iter')
            del self.data_batch
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    def run(self,
            data_loaders: List[DataLoader],
            workflow: List[Tuple[str, int]],
            max_epochs: Optional[int] = None,
            **kwargs) -> None:

        self.grad_clip = None
        _hooks = []
        for hook in self._hooks:
            if isinstance(hook, Fp16OptimizerHook) or \
               isinstance(hook, GradientCumulativeFp16OptimizerHook):
                raise AttributeError('MMCV based FP16 is not supported by %s' % self.__class__.__name__)
            elif isinstance(hook, DistOptimizerHook):
                self.grad_clip = hook.grad_clip
            elif not isinstance(hook, DistOptimizerHook):
                _hooks.append(hook)
        self._hooks = _hooks

        super(AmpEpochBasedRunner, self).run(data_loaders, workflow, max_epochs, **kwargs)


@RUNNERS.register_module()
class AutoAmpEpochBasedRunner(EpochBasedRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler

    def run_iter(self, data_batch: Any, train_mode: bool, **kwargs) -> None:
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer,
                                            **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')

        self.outputs = outputs


    def _amp_train_step(self, data_batch, **kwargs):
        with autocast():
            self.run_iter(data_batch, train_mode=True, **kwargs)
            losses = self.outputs['loss']

        with torch.no_grad():
            loss_naninf = torch.logical_or(losses.isnan(), losses.isinf()).to(dtype=torch.float32)
            loss_naninf_all = logical_or_dist_scalar(loss_naninf)

        if loss_naninf_all:
            dummy_loss = losses * 0
            dummy_loss.backward()
            del self.outputs
            self.model.zero_grad()
            self._fp32_train_step(data_batch, reason=0, **kwargs)
            return

        self.optimizer.zero_grad()
        scaled_loss = self.grad_scaler.scale(losses)

        scaled_loss.backward()
        if self.grad_clip is not None:
            self.grad_scaler.unscale_(self.optimizer)
            _, grad_naninf = clip_grad_norm(self.model.parameters(), self.grad_clip['max_norm'])

            with torch.no_grad():
                grad_naninf_all = logical_or_dist_scalar(grad_naninf.to(dtype=torch.float32))

            if grad_naninf_all:
                self.grad_scaler._per_optimizer_states[id(self.optimizer)]["stage"] = OptState.READY
                self.grad_scaler._per_optimizer_states[id(self.optimizer)]["found_inf_per_device"] = {}
                del self.outputs
                self.model.zero_grad()
                self._fp32_train_step(data_batch, reason=1, **kwargs)
                return
        else:
            raise NotImplementedError("AutoAmpEpochBasedRunner should be used with grad_clip")

        if 'log_vars' in self.outputs:
            self.log_buffer.update(self.outputs['log_vars'], self.outputs['num_samples'])
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()


    def _fp32_train_step(self, data_batch, reason, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            if reason == 0:
                _rs = "during forwarding"
            else:
                _rs = "during backwarding"
            print("AMP failed %s! Turn to FP32! (Step %d)"%(_rs, self.iter))
        torch.cuda.empty_cache()
        self.run_iter(data_batch, train_mode=True, **kwargs)
        losses = self.outputs['loss']
        self.optimizer.zero_grad()
        losses.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip['max_norm'])
        if 'log_vars' in self.outputs:
            self.log_buffer.update(self.outputs['log_vars'], self.outputs['num_samples'])
        self.optimizer.step()

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self.data_batch = data_batch
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self._amp_train_step(data_batch, **kwargs)
            self.call_hook('after_train_iter')
            del self.data_batch
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    def run(self,
            data_loaders: List[DataLoader],
            workflow: List[Tuple[str, int]],
            max_epochs: Optional[int] = None,
            **kwargs) -> None:

        self.grad_clip = None
        _hooks = []
        for hook in self._hooks:
            if isinstance(hook, Fp16OptimizerHook) or \
               isinstance(hook, GradientCumulativeFp16OptimizerHook):
                raise AttributeError('MMCV based FP16 is not supported by %s' % self.__class__.__name__)
            elif isinstance(hook, DistOptimizerHook):
                self.grad_clip = hook.grad_clip
            elif not isinstance(hook, DistOptimizerHook):
                _hooks.append(hook)
        self._hooks = _hooks

        super(AutoAmpEpochBasedRunner, self).run(data_loaders, workflow, max_epochs, **kwargs)



@RUNNERS.register_module()
class AmpIterBasedRunner(IterBasedRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler


    def _amp_train_step(self, data_batch, **kwargs):
        with autocast():
            outputs = self.model.train_step(data_batch, None, **kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('model.train_step() must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
            self.outputs = outputs
            losses = self.outputs['loss']
        self.optimizer.zero_grad()
        self.grad_scaler.scale(losses).backward()
        if self.grad_clip is not None:
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip['max_norm'])
        else:
            raise NotImplementedError("AmpIterBasedRunner should be used with grad_clip")

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._epoch = data_loader.epoch
        data_batch = next(data_loader)
        self.call_hook('before_train_iter')
        self._amp_train_step(data_batch, **kwargs)
        self.call_hook('after_train_iter')
        self._inner_iter += 1
        self._iter += 1

    def run(self,
            data_loaders: List[DataLoader],
            workflow: List[Tuple[str, int]],
            max_epochs: Optional[int] = None,
            **kwargs) -> None:

        self.grad_clip = None

        _hooks = []
        for hook in self._hooks:
            if isinstance(hook, Fp16OptimizerHook) or \
               isinstance(hook, GradientCumulativeFp16OptimizerHook):
                raise AttributeError('MMCV based FP16 is not supported by %s' % self.__class__.__name__)
            elif isinstance(hook, DistOptimizerHook):
                self.grad_clip = hook.grad_clip
            elif not isinstance(hook, DistOptimizerHook):
                _hooks.append(hook)
        self._hooks = _hooks

        super(AmpIterBasedRunner, self).run(data_loaders, workflow, max_epochs, **kwargs)


def clip_grad_norm(
        parameters, max_norm: float, norm_type: float = 2.0,
    ):
    """torch.nn.utils.clip_grad_norm_"""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.)
    device = grads[0].device
    if norm_type == inf:
        norms = [g.detach().abs().max().to(device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
    naninf = torch.logical_or(total_norm.isnan(), total_norm.isinf())
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for g in grads:
        g.detach().mul_(clip_coef_clamped.to(g.device))
    return total_norm, naninf

@torch.no_grad()
def logical_or_dist_scalar(tensor):
    """Reference: MoCo v2"""
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    flag = sum([x.item() for x in tensors_gather])
    return flag

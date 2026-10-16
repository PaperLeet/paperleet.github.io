"""CPU learning-rate schedules and in-memory training continuation state."""
from bisect import bisect_right
from collections import Counter
from functools import wraps
import math
import types
import warnings
import weakref

import numpy as np

from . import _autograd, _errors
from ._optimizer import Optimizer, scalar
from ._tensor import Tensor


def _copy_rate(value):
    if isinstance(value, Tensor):
        if isinstance(value, _autograd.ForwardTensor):
            raise _errors.BrowserUnsupported('Forward differentiation of scheduler learning rates is unsupported')
        scalar(value, 'lr')  # Reject trainable/non-scalar learning rates.
        if type(value) is Tensor:
            return Tensor(value._data.copy(order='K'))
        return value.clone()
    return value


def _rates(optimizer, name='lr'):
    return [_copy_rate(group[name]) for group in optimizer.param_groups]


def _current_rates(optimizer):
    return [group['lr'] for group in optimizer.param_groups]


def _fast_scalar(rate, *coefficients):
    if (type(rate) is Tensor and rate.dtype in (np.dtype('float32'),np.dtype('float64'))
            and all(type(value) in (int,float) for value in coefficients)):
        scalar(rate, 'lr')
        return rate.dtype.type
    return None


def _scaled(rate, factor):
    dtype = _fast_scalar(rate, factor)
    if dtype is not None:
        # Strong NumPy scalars keep every operation in the Tensor's dtype on
        # both NumPy 1.x and 2.x, including float32 rounding of Python scalars.
        return Tensor(rate._data * dtype(factor))
    return rate * factor


def _cosine_rate(rate, ratio, minimum):
    dtype = _fast_scalar(rate, ratio, minimum)
    if dtype is not None:
        return Tensor(dtype(ratio) * (rate._data - dtype(minimum)) + dtype(minimum))
    return ratio * (rate - minimum) + minimum


def _write_rate(group, value):
    if isinstance(group['lr'], Tensor):
        group['lr'].fill_(scalar(value, 'lr'))
    else:
        group['lr'] = value


class LRScheduler:
    _is_initial = False
    _get_lr_called_within_step = False

    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('A scheduler requires an Optimizer')
        self.optimizer = optimizer
        for group in optimizer.param_groups:
            if last_epoch == -1:
                group.setdefault('initial_lr', _copy_rate(group['lr']))
            elif 'initial_lr' not in group:
                raise KeyError('initial_lr is required when resuming a scheduler')
        self.base_lrs = _rates(optimizer, 'initial_lr')
        self.last_epoch = last_epoch
        self._get_lr_called_within_step = False
        self._is_initial = False
        if not getattr(optimizer.step, '_wrapped_by_lr_sched', False):
            reference, function = weakref.ref(optimizer), optimizer.step.__func__
            @wraps(function)
            def tracked_step(*args, **kwargs):
                target = reference()
                target._opt_called = True
                return function(target, *args, **kwargs)
            tracked_step._wrapped_by_lr_sched = True
            optimizer.step = tracked_step
        self._initial_step()

    def _initial_step(self):
        self._step_count = 0
        previous = self._is_initial
        try:
            self._is_initial = True
            self.step()
        finally:
            self._is_initial = previous

    def _update_lr(self, epoch=None):
        previous = self._get_lr_called_within_step
        try:
            self._get_lr_called_within_step = True
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
                self.last_epoch = epoch
                closed = getattr(self, '_get_closed_form_lr', None)
                values = closed() if closed is not None else self.get_lr()
        finally:
            self._get_lr_called_within_step = previous
        for group, value in zip(self.optimizer.param_groups, values, strict=True):
            _write_rate(group, value)
        self._last_lr = _rates(self.optimizer)

    def step(self, epoch=None):
        if self._step_count == 1:
            if not getattr(self.optimizer.step, '_wrapped_by_lr_sched', False):
                warnings.warn('optimizer.step() has been overridden after scheduler initialization', UserWarning, stacklevel=2)
            elif not getattr(self.optimizer, '_opt_called', False):
                warnings.warn('lr_scheduler.step() called before optimizer.step()', UserWarning, stacklevel=2)
        self._step_count += 1
        if epoch is not None:
            warnings.warn('The epoch parameter in scheduler.step() is deprecated', UserWarning, stacklevel=2)
        self._update_lr(epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn('Use get_last_lr() to inspect the most recent learning rate', UserWarning, stacklevel=2)
        return self._next_rates()

    def _next_rates(self):
        raise NotImplementedError

    def get_last_lr(self):
        return self._last_lr

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


class _LRScheduler(LRScheduler):
    pass


class StepLR(LRScheduler):
    def __init__(self, optimizer, step_size, gamma=.1, last_epoch=-1):
        self.step_size, self.gamma = step_size, gamma
        super().__init__(optimizer, last_epoch)

    def _next_rates(self):
        if self.last_epoch == 0 or self.last_epoch % self.step_size:
            return _rates(self.optimizer)
        return [_scaled(rate, self.gamma) for rate in _current_rates(self.optimizer)]

    def _get_closed_form_lr(self):
        return [rate * self.gamma ** (self.last_epoch // self.step_size) for rate in self.base_lrs]


class MultiStepLR(LRScheduler):
    def __init__(self, optimizer, milestones, gamma=.1, last_epoch=-1):
        self.milestones, self.gamma = Counter(milestones), gamma
        super().__init__(optimizer, last_epoch)

    def _next_rates(self):
        count = self.milestones[self.last_epoch]
        if not count:
            return _rates(self.optimizer)
        return [_scaled(rate, self.gamma ** count) for rate in _current_rates(self.optimizer)]

    def _get_closed_form_lr(self):
        count = sum(count for epoch, count in self.milestones.items() if epoch <= self.last_epoch)
        return [rate * self.gamma ** count for rate in self.base_lrs]


class ExponentialLR(LRScheduler):
    def __init__(self, optimizer, gamma, last_epoch=-1):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def _next_rates(self):
        if self._is_initial:
            return _rates(self.optimizer)
        return [_scaled(rate, self.gamma) for rate in _current_rates(self.optimizer)]

    def _get_closed_form_lr(self):
        return [rate * self.gamma ** self.last_epoch for rate in self.base_lrs]


class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0., last_epoch=-1):
        self.T_max, self.eta_min = T_max, eta_min
        super().__init__(optimizer, last_epoch)

    def _next_rates(self):
        if self._is_initial:
            return _rates(self.optimizer)
        if self._step_count == 1 and self.last_epoch > 0:
            return self._get_closed_form_lr()
        angle = math.pi / self.T_max
        if (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [rate + (base - self.eta_min) * (1 - math.cos(angle)) / 2
                    for rate, base in zip(_current_rates(self.optimizer), self.base_lrs, strict=True)]
        ratio = (1 + math.cos(angle * self.last_epoch)) / (1 + math.cos(angle * (self.last_epoch - 1)))
        return [_cosine_rate(rate, ratio, self.eta_min) for rate in _current_rates(self.optimizer)]

    def _get_closed_form_lr(self):
        factor = (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
        return [self.eta_min + (rate - self.eta_min) * factor for rate in self.base_lrs]


class LinearLR(LRScheduler):
    def __init__(self, optimizer, start_factor=1/3, end_factor=1., total_iters=5, last_epoch=-1):
        if not 0 < start_factor <= 1 or not 0 <= end_factor <= 1:
            raise ValueError('LinearLR factors must have start in (0,1] and end in [0,1]')
        self.start_factor, self.end_factor, self.total_iters = start_factor, end_factor, total_iters
        super().__init__(optimizer, last_epoch)

    def _next_rates(self):
        if self.last_epoch == 0:
            return [_scaled(rate, self.start_factor) for rate in _current_rates(self.optimizer)]
        if self._is_initial or self.last_epoch > self.total_iters:
            return _rates(self.optimizer)
        delta = self.end_factor - self.start_factor
        factor = 1 + delta / (self.total_iters * self.start_factor + (self.last_epoch - 1) * delta)
        return [_scaled(rate, factor) for rate in _current_rates(self.optimizer)]

    def _get_closed_form_lr(self):
        factor = self.start_factor + (self.end_factor - self.start_factor) * min(self.last_epoch, self.total_iters) / self.total_iters
        return [rate * factor for rate in self.base_lrs]


class ConstantLR(LRScheduler):
    def __init__(self, optimizer, factor=1/3, total_iters=5, last_epoch=-1):
        if not 0 <= factor <= 1:
            raise ValueError('ConstantLR factor must be in [0,1]')
        self.factor, self.total_iters = factor, total_iters
        super().__init__(optimizer, last_epoch)

    def _next_rates(self):
        if self.last_epoch != 0 and self.last_epoch != self.total_iters:
            return _rates(self.optimizer)
        factor = self.factor if self.last_epoch == 0 else 1 / self.factor
        return [_scaled(rate, factor) for rate in _current_rates(self.optimizer)]

    def _get_closed_form_lr(self):
        factor = self.factor if self.last_epoch < self.total_iters else 1
        return [rate * factor for rate in self.base_lrs]


class PolynomialLR(LRScheduler):
    def __init__(self, optimizer, total_iters=5, power=1., last_epoch=-1):
        self.total_iters, self.power = total_iters, power
        super().__init__(optimizer, last_epoch)

    def _next_rates(self):
        if self._is_initial or self.last_epoch > self.total_iters:
            return _rates(self.optimizer)
        factor = ((1 - self.last_epoch / self.total_iters) /
                  (1 - (self.last_epoch - 1) / self.total_iters)) ** self.power
        return [_scaled(rate, factor) for rate in _current_rates(self.optimizer)]

    def _get_closed_form_lr(self):
        factor = (1 - min(self.last_epoch, self.total_iters) / self.total_iters) ** self.power
        return [rate * factor for rate in self.base_lrs]


class LambdaLR(LRScheduler):
    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        functions = list(lr_lambda) if isinstance(lr_lambda, (list, tuple)) else [lr_lambda] * len(optimizer.param_groups)
        if len(functions) != len(optimizer.param_groups):
            raise ValueError('Expected one lr_lambda per parameter group')
        self.lr_lambdas = functions
        super().__init__(optimizer, last_epoch)

    def _next_rates(self):
        return [_scaled(base, function(self.last_epoch)) for base, function in zip(self.base_lrs, self.lr_lambdas, strict=True)]

    def state_dict(self):
        state = super().state_dict()
        state['lr_lambdas'] = [None if isinstance(function, types.FunctionType) else function.__dict__.copy()
                               for function in self.lr_lambdas]
        return state

    def load_state_dict(self, state_dict):
        state = dict(state_dict)
        functions = state.pop('lr_lambdas')
        super().load_state_dict(state)
        for index, attributes in enumerate(functions):
            if attributes is not None:
                self.lr_lambdas[index].__dict__.update(attributes)


class MultiplicativeLR(LambdaLR):
    def _next_rates(self):
        if self._is_initial:
            return _rates(self.optimizer)
        return [_scaled(rate, function(self.last_epoch)) for rate, function in zip(_current_rates(self.optimizer), self.lr_lambdas, strict=True)]


def _check_children(optimizer, schedulers):
    if not schedulers:
        raise ValueError('A composite scheduler needs at least one child')
    if any(not isinstance(s, LRScheduler) for s in schedulers):
        raise TypeError('Children must be learning-rate schedulers')
    if any(isinstance(s, ReduceLROnPlateau) for s in schedulers):
        raise ValueError('ReduceLROnPlateau requires a metric and cannot be composed')
    if any(s.optimizer is not optimizer for s in schedulers):
        raise ValueError('Scheduler children must share the same optimizer')


class _CompositeScheduler(LRScheduler):
    def state_dict(self):
        state = super().state_dict()
        state['_schedulers'] = [scheduler.state_dict() for scheduler in self._schedulers]
        return state

    def load_state_dict(self, state_dict):
        state = dict(state_dict)
        children = state.pop('_schedulers')
        super().load_state_dict(state)
        for index, saved in enumerate(children):
            self._schedulers[index].load_state_dict(saved)


class ChainedScheduler(_CompositeScheduler):
    def __init__(self, schedulers, optimizer=None):
        if not schedulers:
            raise ValueError('ChainedScheduler needs at least one child')
        optimizer = schedulers[0].optimizer if optimizer is None else optimizer
        _check_children(optimizer, schedulers)
        self.optimizer, self._schedulers = optimizer, list(schedulers)
        self._last_lr = _rates(optimizer)

    def step(self):
        for scheduler in self._schedulers:
            scheduler.step()
        self._last_lr = _rates(self.optimizer)


class SequentialLR(_CompositeScheduler):
    def __init__(self, optimizer, schedulers, milestones, last_epoch=-1):
        _check_children(optimizer, schedulers)
        if len(milestones) != len(schedulers) - 1:
            raise ValueError('SequentialLR requires one fewer milestone than schedulers')
        self.optimizer, self._schedulers, self._milestones = optimizer, list(schedulers), list(milestones)
        self.last_epoch = last_epoch + 1
        for group in optimizer.param_groups:
            _write_rate(group, group['initial_lr'])
        self.recursive_undo()
        active = self._schedulers[bisect_right(self._milestones, 0)]
        active._initial_step()
        self._last_lr = active.get_last_lr()

    def recursive_undo(self, sched=None):
        current = self if sched is None else sched
        if isinstance(current, _CompositeScheduler):
            for child in current._schedulers:
                self.recursive_undo(child)
        else:
            current.last_epoch -= 1

    def step(self):
        self.last_epoch += 1
        index = bisect_right(self._milestones, self.last_epoch)
        active = self._schedulers[index]
        if index and self.last_epoch == self._milestones[index - 1]:
            active._update_lr(0)
        else:
            active.step()
        self._last_lr = active.get_last_lr()


class ReduceLROnPlateau(LRScheduler):
    def __init__(self, optimizer, mode='min', factor=.1, patience=10, threshold=1e-4,
                 threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8):
        if factor >= 1:
            raise ValueError('Plateau factor must be less than one')
        if not isinstance(optimizer, Optimizer):
            raise TypeError('A scheduler requires an Optimizer')
        self.optimizer, self.factor = optimizer, factor
        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError('Expected one min_lr per parameter group')
            self.min_lrs, self.default_min_lr = list(min_lr), None
        else:
            self.default_min_lr = min_lr
            self.min_lrs = [min_lr] * len(optimizer.param_groups)
        self.patience, self.cooldown, self.eps = patience, cooldown, eps
        self.last_epoch, self._last_lr = 0, _rates(optimizer)
        self._init_is_better(mode, threshold, threshold_mode)
        self._reset()

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in ('min', 'max') or threshold_mode not in ('rel', 'abs'):
            raise ValueError('Unknown plateau mode or threshold_mode')
        self.mode, self.threshold, self.threshold_mode = mode, threshold, threshold_mode
        self.mode_worse = math.inf if mode == 'min' else -math.inf

    def _reset(self):
        self.best, self.cooldown_counter, self.num_bad_epochs = self.mode_worse, 0, 0

    def _is_better(self, value, best):
        if self.mode == 'min':
            boundary = best - self.threshold if self.threshold_mode == 'abs' else best * (1 - self.threshold)
            return value < boundary
        boundary = best + self.threshold if self.threshold_mode == 'abs' else best * (1 + self.threshold)
        return value > boundary

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def _reduce_lr(self, epoch):
        if len(self.min_lrs) != len(self.optimizer.param_groups):
            if self.default_min_lr is None:
                raise RuntimeError('Update min_lrs after adding optimizer parameter groups')
            self.min_lrs = [self.default_min_lr] * len(self.optimizer.param_groups)
        for group, minimum in zip(self.optimizer.param_groups, self.min_lrs):
            old = scalar(group['lr'], 'lr')
            value = max(old * self.factor, minimum)
            if old - value > self.eps:
                _write_rate(group, value)

    def step(self, metrics, epoch=None):
        current = float(metrics)
        if epoch is not None:
            warnings.warn('The epoch parameter in scheduler.step() is deprecated', UserWarning, stacklevel=2)
        self.last_epoch = self.last_epoch + 1 if epoch is None else epoch
        if self._is_better(current, self.best):
            self.best, self.num_bad_epochs = current, 0
        else:
            self.num_bad_epochs += 1
        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0
        if self.num_bad_epochs > self.patience:
            self._reduce_lr(self.last_epoch)
            self.cooldown_counter, self.num_bad_epochs = self.cooldown, 0
        self._last_lr = _rates(self.optimizer)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._init_is_better(self.mode, self.threshold, self.threshold_mode)

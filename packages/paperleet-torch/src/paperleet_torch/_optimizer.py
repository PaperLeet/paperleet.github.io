"""Shared optimizer groups, gradient buffers and in-memory checkpoints."""
from collections import defaultdict
from collections.abc import Mapping
import copy
import numbers
import warnings

import numpy as np

from . import _errors, _state, _tensor


def scalar(value, name):
    if type(value) is float or type(value) is int:
        return value
    if isinstance(value, _tensor.Tensor):
        if value.numel() != 1:
            raise ValueError(f'{name} must have one element')
        if value.requires_grad or value._parents:
            raise _errors.BrowserUnsupported(f'Differentiable optimizer {name} is unsupported')
        value = value.item()
    if not isinstance(value, numbers.Real):
        raise TypeError(f'{name} must be a real number')
    return float(value)


def nonnegative(value, name):
    value = scalar(value, name)
    if not value >= 0:
        raise ValueError(f'{name} must be nonnegative')
    return value


def group_option(name):
    """Retain the original adapter's single-group attribute spelling."""
    return property(lambda self: self.param_groups[0][name],
                    lambda self, value: self.param_groups[0].__setitem__(name, value))


class Optimizer:
    """CPU float32/float64 optimizer lifecycle, independent of training kernels.

    Hook registration, differentiable updates and accelerator execution options
    fail explicitly. Checkpoints use positional parameter indices as PyTorch does.
    """
    lr = group_option('lr')

    def __init__(self, params, defaults):
        self.defaults = dict(defaults)
        self.state = defaultdict(dict)
        self.param_groups = []
        self._validate_defaults(self.defaults)
        if isinstance(params, _tensor.Tensor):
            raise TypeError('Optimizer parameters require an ordered iterable')
        groups = list(params)
        if not groups:
            raise ValueError('Optimizer received an empty parameter list')
        if not isinstance(groups[0], dict):
            groups = [{'params': groups}]
        for group in groups:
            self.add_param_group(group)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return _errors.unsupported(f'{type(self).__name__}.{name}')

    @property
    def params(self):
        if len(self.param_groups) == 1:
            return self.param_groups[0]['params']
        return [p for group in self.param_groups for p in group['params']]

    def _validate_group(self, group):
        for option in ('foreach', 'fused', 'capturable', 'differentiable'):
            if group.get(option) not in (None, False):
                raise _errors.BrowserUnsupported(f'Optimizer {option}=True is unsupported on NumPy')

    def _validate_defaults(self, defaults):
        pass

    def add_param_group(self, param_group):
        if not isinstance(param_group, dict):
            raise TypeError('param_group must be a dict')
        params = param_group['params']
        if isinstance(params, _tensor.Tensor):
            params = [params]
        elif isinstance(params, set):
            raise TypeError('Optimizer parameters require an ordered collection')
        else:
            params = list(params)
        names, tensors = [], []
        for param in params:
            if isinstance(param, tuple):
                if len(param) != 2 or not isinstance(param[0], str):
                    raise TypeError('Named optimizer parameters require (name, Tensor) pairs')
                name, param = param
                names.append(name)
            tensors.append(param)
        if names:
            if len(names) != len(tensors):
                raise ValueError('All optimizer parameters must be named or unnamed')
            param_group['param_names'] = names
        for param in tensors:
            if not isinstance(param, _tensor.Tensor):
                raise TypeError('Optimizer can only optimize Tensors')
            if param._parents:
                raise ValueError('Cannot optimize a non-leaf Tensor')
            if param.dtype not in (np.dtype('float32'), np.dtype('float64')):
                raise _errors.BrowserUnsupported('Optimizers require float32/float64 parameters')
        for name, default in self.defaults.items():
            param_group.setdefault(name, default)
        self._validate_group(param_group)
        ids = {id(param) for param in tensors}
        if len(ids) != len(tensors):
            warnings.warn('Optimizer parameter group contains duplicate parameters', UserWarning, stacklevel=2)
        for group in self.param_groups:
            if ('param_names' in group) != ('param_names' in param_group):
                raise ValueError('All optimizer groups must be named or unnamed')
            if ids.intersection(id(param) for param in group['params']):
                raise ValueError('Parameters appear in more than one parameter group')
        param_group['params'] = tensors
        self.param_groups.append(param_group)

    def zero_grad(self, set_to_none=True):
        for group in self.param_groups:
            for parameter in group['params']:
                grad = parameter.grad
                if grad is None:
                    continue
                if set_to_none:
                    parameter.grad = None
                else:
                    if grad._parents:
                        # Higher-order gradient construction is outside this backend.
                        raise _errors.BrowserUnsupported('Clearing differentiable gradients is unsupported')
                    if grad.requires_grad:
                        grad.requires_grad_(False)
                    grad.zero_()

    def _start_step(self, closure):
        if _state._jvp_active:
            raise _errors.BrowserUnsupported('Optimizer mutation during JVP is unsupported')
        if closure is None:
            return None
        previous = _state._grad_enabled
        try:
            _state._grad_enabled = True
            return closure()
        finally:
            _state._grad_enabled = previous

    @staticmethod
    def _gradient(parameter, *, allow_mixed=False):
        grad = parameter.grad
        if not isinstance(grad, _tensor.Tensor) or grad.shape != parameter.shape:
            raise RuntimeError('Optimizer gradient shape/dtype must match the parameter')
        if grad.dtype != parameter.dtype and not (allow_mixed and grad.dtype in (np.dtype('float32'), np.dtype('float64'))):
            raise RuntimeError('Optimizer gradient shape/dtype must match the parameter')
        return grad._data

    @staticmethod
    def _buffer(value, parameter, name, *, step=False, allow_mixed=False):
        if not isinstance(value, _tensor.Tensor):
            raise RuntimeError(f'Optimizer state {name} must be a Tensor')
        if value._parents:
            raise _errors.BrowserUnsupported('Optimizer state with a gradient graph is unsupported')
        if step:
            if value.numel() != 1 or value.dtype.kind != 'f':
                raise RuntimeError('Optimizer step state must be a floating scalar')
        elif value.shape != parameter.shape or value.dtype != parameter.dtype and not (allow_mixed and value.dtype in (np.dtype('float32'), np.dtype('float64'))):
            raise RuntimeError(f'Optimizer state {name} shape/dtype must match the parameter')
        return value._data

    def state_dict(self):
        mapping, packed_groups, start = {}, [], 0
        for group in self.param_groups:
            packed = {key: value for key, value in group.items() if key != 'params'}
            # Duplicate entries in one group use their final positional index.
            mapping.update({id(param): index for index, param in enumerate(group['params'], start)
                            if id(param) not in mapping})
            packed['params'] = [mapping[id(param)] for param in group['params']]
            start += len(group['params'])
            packed_groups.append(packed)
        state = {(mapping[id(key)] if isinstance(key, _tensor.Tensor) else key): value
                 for key, value in self.state.items()}
        return {'state': state, 'param_groups': packed_groups}

    def load_state_dict(self, state_dict):
        if not isinstance(state_dict, Mapping):
            raise TypeError('Optimizer state_dict must be a mapping')
        saved = copy.deepcopy(state_dict['param_groups'])
        if len(saved) != len(self.param_groups):
            raise ValueError('Loaded optimizer has a different number of parameter groups')
        if any(len(old['params']) != len(new['params']) for old, new in zip(self.param_groups, saved)):
            raise ValueError('Loaded optimizer parameter group has a different size')
        mapping, new_groups = {}, []
        for current, group in zip(self.param_groups, saved):
            mapping.update(zip(group['params'], current['params']))
            group['params'] = current['params']
            if 'param_names' in current and 'param_names' not in group:
                group['param_names'] = current['param_names']
            # Accept older checkpoints lacking optional implementation flags.
            for key, value in self.defaults.items():
                group.setdefault(key, value)
            self._validate_group(group)
            new_groups.append(group)

        def cast(value, parameter, key=None):
            if isinstance(value, _tensor.Tensor):
                # Non-capturable step counters retain their checkpoint dtype;
                # moments follow the destination parameter. Same-dtype tensors
                # remain shared with the supplied checkpoint, as in native .to.
                return value if key == 'step' else value.to(dtype=parameter.dtype)
            if isinstance(value, dict):
                return {name: cast(item, parameter, name) for name, item in value.items()}
            if isinstance(value, (list, tuple)):
                return type(value)(cast(item, parameter) for item in value)
            return value

        state = defaultdict(dict)
        for key, value in state_dict['state'].items():
            target = mapping[key] if key in mapping else key
            state[target] = cast(value, target) if key in mapping else value
        self.state, self.param_groups = state, new_groups
        self._after_load()

    def _after_load(self):
        pass

    def step(self, closure=None):
        raise NotImplementedError

    def __repr__(self):
        groups = []
        for index, group in enumerate(self.param_groups):
            values = '\n'.join(f'    {name}: {group[name]}' for name in sorted(group) if name != 'params')
            groups.append(f'Parameter Group {index}\n{values}')
        return type(self).__name__ + ' (\n' + '\n\n'.join(groups) + '\n)'

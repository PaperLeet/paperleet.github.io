"""Raw NumPy SGD/Adam kernels with shared model-independent lifecycle."""
import math
import numpy as np

from . import _errors, _tensor
from ._optimizer import Optimizer, group_option, nonnegative, scalar


class SGD(Optimizer):
    momentum = group_option('momentum')
    dampening = group_option('dampening')
    weight_decay = group_option('weight_decay')
    nesterov = group_option('nesterov')
    maximize = group_option('maximize')

    def __init__(self, params, lr=.001, momentum=0, dampening=0, weight_decay=0, nesterov=False,
                 *, maximize=False, foreach=None, differentiable=False, fused=None, **options):
        if options:
            raise _errors.BrowserUnsupported('Unknown SGD options: ' + ', '.join(options))
        super().__init__(params, dict(lr=lr, momentum=momentum, dampening=dampening,
                                     weight_decay=weight_decay, nesterov=nesterov, maximize=maximize,
                                     foreach=foreach, differentiable=differentiable, fused=fused))

    def _validate_group(self, group):
        super()._validate_group(group)
        return self._validate_defaults(group)

    def _validate_defaults(self, group):
        lr, momentum, weight_decay = [scalar(group[name], name) for name in ('lr', 'momentum', 'weight_decay')]
        for name, value in (('lr', lr), ('momentum', momentum), ('weight_decay', weight_decay)):
            if value < 0:
                raise ValueError(f'{name} must be nonnegative')
        dampening = scalar(group['dampening'], 'dampening')
        if group['nesterov'] and (momentum <= 0 or dampening != 0):
            raise ValueError('Nesterov requires momentum and zero dampening')
        return lr, momentum, weight_decay, dampening

    def step(self, closure=None):
        loss = self._start_step(closure)
        for group in self.param_groups:
            lr, momentum, weight_decay, dampening = self._validate_group(group)
            # Capture initial buffers before mutation, preserving native behavior
            # when a warned-about duplicate first acquires momentum in this call.
            active = ((p, None) for p in group['params'] if p.grad is not None)
            if momentum:
                active = [(p, self.state[p].get('momentum_buffer')) for p in group['params'] if p.grad is not None]
            for parameter, buffer in active:
                gradient = self._gradient(parameter, allow_mixed=True)
                if group['maximize']:
                    gradient = -gradient
                if weight_decay:
                    gradient = gradient + weight_decay * parameter._data
                if momentum:
                    if buffer is None:
                        buffer = _tensor.Tensor(gradient.copy())
                    else:
                        value = self._buffer(buffer, parameter, 'momentum_buffer', allow_mixed=True)
                        value *= momentum
                        value += (1 - dampening) * gradient
                        buffer._storage.bump()
                    self.state[parameter]['momentum_buffer'] = buffer
                    gradient = gradient + momentum * buffer._data if group['nesterov'] else buffer._data
                if gradient.dtype != parameter.dtype:
                    gradient = gradient.astype(np.result_type(gradient.dtype, parameter.dtype), copy=False)
                parameter._data[...] -= lr * gradient
                parameter._storage.bump()
        return loss


class Adam(Optimizer):
    betas = group_option('betas')
    eps = group_option('eps')
    weight_decay = group_option('weight_decay')

    def __init__(self, params, lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0, amsgrad=False,
                 *, foreach=None, maximize=False, capturable=False, differentiable=False, fused=None,
                 decoupled_weight_decay=False, **options):
        if options:
            raise _errors.BrowserUnsupported('Unknown Adam options: ' + ', '.join(options))
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                     amsgrad=amsgrad, maximize=maximize, foreach=foreach,
                                     capturable=capturable, differentiable=differentiable, fused=fused,
                                     decoupled_weight_decay=decoupled_weight_decay))

    def _validate_group(self, group):
        super()._validate_group(group)
        return self._validate_defaults(group)

    def _validate_defaults(self, group):
        lr, eps, weight_decay = [nonnegative(group[name], name) for name in ('lr', 'eps', 'weight_decay')]
        betas = group['betas']
        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise ValueError('Adam betas must contain two coefficients')
        b1, b2 = [scalar(value, 'beta') for value in betas]
        if not 0 <= b1 < 1 or not 0 <= b2 < 1:
            raise ValueError('Adam beta coefficients must be in [0, 1)')
        return lr, eps, weight_decay, b1, b2

    def _after_load(self):
        for group in self.param_groups:
            for parameter in group['params']:
                value = self.state.get(parameter)
                if value and 'step' in value and not isinstance(value['step'], _tensor.Tensor):
                    value['step'] = _tensor.Tensor(np.array(float(value['step']), dtype=np.float32))

    def step(self, closure=None):
        loss = self._start_step(closure)
        for group in self.param_groups:
            lr, eps, weight_decay, b1, b2 = self._validate_group(group)
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                gradient = self._gradient(parameter)
                if group['maximize']:
                    gradient = -gradient
                state = self.state[parameter]
                if not state:
                    state.update(step=_tensor.Tensor(np.array(0., dtype=np.float32)),
                                 exp_avg=_tensor.Tensor(np.zeros_like(parameter._data)),
                                 exp_avg_sq=_tensor.Tensor(np.zeros_like(parameter._data)))
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = _tensor.Tensor(np.zeros_like(parameter._data))
                count = self._buffer(state['step'], parameter, 'step', step=True)
                moment = self._buffer(state['exp_avg'], parameter, 'exp_avg')
                variance = self._buffer(state['exp_avg_sq'], parameter, 'exp_avg_sq')
                maximum = self._buffer(state['max_exp_avg_sq'], parameter, 'max_exp_avg_sq') if group['amsgrad'] else None
                count[...] += 1
                step = float(count.item())
                if weight_decay:
                    if group['decoupled_weight_decay']:
                        parameter._data[...] *= 1 - lr * weight_decay
                    else:
                        gradient = gradient + weight_decay * parameter._data
                # Independent NumPy oracle: do not reuse learner Tensor updates.
                moment += (1 - b1) * (gradient - moment)
                variance *= b2
                variance += (1 - b2) * gradient * gradient
                if maximum is not None:
                    np.maximum(maximum, variance, out=maximum)
                denominator = np.sqrt(variance if maximum is None else maximum) / math.sqrt(1 - b2 ** step) + eps
                parameter._data[...] -= (lr / (1 - b1 ** step)) * moment / denominator
                parameter._storage.bump()
                for name in ('step', 'exp_avg', 'exp_avg_sq', *(['max_exp_avg_sq'] if maximum is not None else [])):
                    state[name]._storage.bump()
        return loss


class AdamW(Adam):
    """Decoupled Adam with the same independently evaluated NumPy oracle."""
    def __init__(self, params, lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=.01, amsgrad=False,
                 *, maximize=False, foreach=None, capturable=False, differentiable=False, fused=None, **options):
        if options:
            raise _errors.BrowserUnsupported('Unknown AdamW options: ' + ', '.join(options))
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad,
                         maximize=maximize, foreach=foreach, capturable=capturable,
                         differentiable=differentiable, fused=fused, decoupled_weight_decay=True)

    def _after_load(self):
        super()._after_load()
        for group in self.param_groups:
            group['decoupled_weight_decay'] = True

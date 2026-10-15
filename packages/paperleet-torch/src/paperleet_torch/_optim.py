"""NumPy-backed optim implementation; extracted from PaperLeet."""
import numpy as np


class SGD:
    def __init__(self, params, lr=.001, momentum=0, dampening=0, weight_decay=0, nesterov=False, *, maximize=False, **options):
        if options: raise _errors.BrowserUnsupported('Advanced SGD options require the server')
        if lr < 0 or momentum < 0 or weight_decay < 0: raise ValueError('Invalid SGD hyperparameter')
        if nesterov and (momentum <= 0 or dampening != 0): raise ValueError('Nesterov requires momentum and zero dampening')
        self.params, self.lr, self.momentum = list(params), lr, momentum
        if any(not isinstance(p, _tensor.Tensor) for p in self.params): raise _errors.BrowserUnsupported('SGD parameter groups require the server')
        self.dampening, self.weight_decay, self.nesterov, self.maximize = dampening, weight_decay, nesterov, maximize
        self.buffers = [None] * len(self.params)

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            if p.grad is not None: p.grad = None if set_to_none else _tensor.Tensor(np.zeros_like(p._data))

    def step(self, closure=None):
        if closure is not None: raise _errors.BrowserUnsupported('SGD closures require the server')
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            g = p.grad._data.copy()
            if self.maximize: g = -g
            if self.weight_decay: g += self.weight_decay * p._data
            if self.momentum:
                buffer = self.buffers[i]
                if buffer is None: buffer = g.copy()
                else: buffer = self.momentum * buffer + (1 - self.dampening) * g
                self.buffers[i] = buffer
                g = g + self.momentum * buffer if self.nesterov else buffer
            p._data[...] -= self.lr * g
            p._storage.version += 1


class Adam:
    def __getattr__(self, name):
        if name.startswith('_'): raise AttributeError(name)
        return _errors.unsupported(f'Adam.{name}')

    def __init__(self, params, lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0):
        if weight_decay: raise _errors.BrowserUnsupported('Adam weight decay requires the server')
        self.params, self.lr, self.betas, self.eps = list(params), lr, betas, eps
        self.steps = 0
        self.m = [np.zeros_like(p._data) for p in self.params]
        self.v = [np.zeros_like(p._data) for p in self.params]
    def zero_grad(self, set_to_none=True):
        for p in self.params: p.grad = None if set_to_none else _tensor.Tensor(np.zeros_like(p._data))
    def step(self):
        self.steps += 1
        b1, b2 = self.betas
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            g = p.grad._data
            self.m[i] *= b1; self.m[i] += (1 - b1) * g
            self.v[i] *= b2; self.v[i] += (1 - b2) * g * g
            p._data[...] -= self.lr * (self.m[i] / (1 - b1 ** self.steps)) / (np.sqrt(self.v[i] / (1 - b2 ** self.steps)) + self.eps)
            p._storage.version += 1


class AdamW:
    """Independent, stateful NumPy oracle for the functional AdamW exercise."""
    def __getattr__(self, name):
        if name.startswith('_'): raise AttributeError(name)
        return _errors.unsupported(f'AdamW.{name}')

    def __init__(self, params, lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=.01, **options):
        if options: raise _errors.BrowserUnsupported('Advanced AdamW options require the server')
        params = list(params)
        if not params: raise ValueError('AdamW needs parameters')
        if any(not isinstance(p, _tensor.Tensor) for p in params):
            raise _errors.BrowserUnsupported('AdamW parameter groups require the server')
        if lr < 0 or eps < 0 or weight_decay < 0 or len(betas) != 2 or any(not 0 <= b < 1 for b in betas):
            raise ValueError('Invalid AdamW options')
        self.param_groups = [dict(params=params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)]
        self.state = {}

    def zero_grad(self, set_to_none=True):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad = None if set_to_none else _tensor.Tensor(np.zeros_like(p._data))

    def step(self, closure=None):
        if closure is not None: raise _errors.BrowserUnsupported('AdamW closures require the server')
        for group in self.param_groups:
            b1, b2 = group['betas']
            for p in group['params']:
                if p.grad is None: continue
                if p.grad.shape != p.shape or p.grad.dtype != p.dtype:
                    raise RuntimeError('AdamW gradient shape/dtype must match the parameter')
                if p not in self.state:
                    self.state[p] = dict(step=_tensor.Tensor(np.array(0., dtype=np.float32)),
                                         exp_avg=_tensor.Tensor(np.zeros_like(p._data)), exp_avg_sq=_tensor.Tensor(np.zeros_like(p._data)))
                state = self.state[p]
                state['step']._data[...] += 1
                count = state['step'].item()
                m, v, g = state['exp_avg']._data, state['exp_avg_sq']._data, p.grad._data
                # Update moments directly, independently of learner Tensor ops.
                m += (1 - b1) * (g - m)
                v *= b2
                v += (1 - b2) * g * g
                p._data[...] *= 1 - group['lr'] * group['weight_decay']
                denominator = np.sqrt(v) / np.sqrt(1 - b2 ** count) + group['eps']
                p._data[...] -= (group['lr'] / (1 - b1 ** count)) * m / denominator
                p._storage.version += 1
                for value in state.values(): value._storage.version += 1


# Late module references break Tensor/autograd/operator cycles without rebinding functions.
from . import _errors, _tensor

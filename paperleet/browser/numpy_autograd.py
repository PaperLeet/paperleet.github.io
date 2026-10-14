"""Bounded NumPy reverse-mode autodiff backend for validated teaching modules.

CPU/WASM only. Gradients are analytical vector-Jacobian products, not numerical
approximations. This is a teaching-subset adapter, not a native PyTorch runtime.
Unsupported APIs raise BrowserUnsupported and retain server judging.
"""
import contextlib
import types
import sys
from collections import namedtuple
import numpy as np


class BrowserUnsupported(NotImplementedError):
    pass


_grad_enabled = True


def unsupported(name):
    if name.startswith('_'):
        raise AttributeError(name)
    raise BrowserUnsupported(f'{name} is outside the CPU/WASM autodiff backend')


def unbroadcast(value, shape):
    while value.ndim > len(shape):
        value = value.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and value.shape[axis] != 1:
            value = value.sum(axis=axis, keepdims=True)
    return value.reshape(shape)


def wrap(value):
    return value if isinstance(value, Tensor) else tensor(value)


def result(data, *edges):
    value = Tensor(data)
    value._parents = tuple((node, vjp) for node, vjp in edges if node.requires_grad and _grad_enabled)
    value.requires_grad = bool(value._parents)
    return value


def backprop(output, seed):
    if not output.requires_grad:
        raise RuntimeError('Tensor does not require grad and has no grad_fn')
    order, visited, stack = [], set(), [(output, False)]
    while stack:
        node, expanded = stack.pop()
        if expanded:
            order.append(node)
        elif id(node) not in visited:
            visited.add(id(node))
            stack.append((node, True))
            stack.extend((parent, False) for parent, _ in node._parents)
    gradients = {id(output): np.asarray(seed, dtype=output.dtype)}
    for node in reversed(order):
        grad = gradients.get(id(node))
        if grad is None:
            continue
        for parent, vjp in node._parents:
            incoming = np.asarray(vjp(grad), dtype=parent.dtype)
            key = id(parent)
            gradients[key] = incoming if key not in gradients else gradients[key] + incoming
    return order, gradients


class Tensor:
    __array_priority__ = 1000
    __hash__ = object.__hash__

    def __init__(self, data, dtype=None, requires_grad=False):
        self._data = np.asarray(data, dtype=dtype)
        self.requires_grad = bool(requires_grad)
        if requires_grad and self.dtype.kind != 'f':
            raise RuntimeError('Only floating point tensors can require gradients')
        self._parents = ()
        self.grad = None

    shape = property(lambda self: self._data.shape)
    ndim = property(lambda self: self._data.ndim)
    dtype = property(lambda self: self._data.dtype)
    device = property(lambda self: Device('cpu'))
    T = property(lambda self: self.permute(*reversed(range(self.ndim))))
    data = property(lambda self: self.detach())

    def __getattr__(self, name):
        if name.startswith('_'): raise AttributeError(name)
        return unsupported(f'Tensor.{name}')
    def __array__(self, dtype=None, copy=None): return np.array(self._data, dtype=dtype, copy=copy) if copy is not None else np.asarray(self._data, dtype=dtype)
    def __len__(self): return len(self._data)
    def __float__(self): return float(self._data)
    def __int__(self): return int(self._data)
    def __bool__(self): return bool(self._data)
    def __repr__(self): return f'Tensor({self._data!r}, requires_grad={self.requires_grad})'
    def item(self): return self._data.item()
    def tolist(self): return self._data.tolist()
    def numpy(self): return self._data
    def numel(self): return self._data.size
    def dim(self): return self.ndim
    def size(self, dim=None): return self.shape if dim is None else self.shape[dim]
    def detach(self): return Tensor(self._data)
    def clone(self): return result(self._data.copy(), (self, lambda g: g))
    def cpu(self): return self
    def contiguous(self): return self.clone()
    def is_contiguous(self): return bool(self._data.flags.c_contiguous)
    def double(self): return self.to(dtype=np.float64)
    def float(self): return self.to(dtype=np.float32)
    def type_as(self, other): return self.to(other)

    def to(self, other=None, *, dtype=None, device=None):
        if isinstance(other, Tensor): dtype = other.dtype
        elif isinstance(other, str) and other.startswith('cpu'): device = other
        elif other is not None: dtype = other
        if device not in (None, 'cpu', 'cpu:0'):
            raise BrowserUnsupported('Only CPU/WASM tensors are supported')
        return result(self._data.astype(dtype or self.dtype, copy=False), (self, lambda g: g.astype(self.dtype)))

    def requires_grad_(self, requires_grad=True):
        if requires_grad and self.dtype.kind != 'f': raise RuntimeError('Only floating tensors can require grad')
        self.requires_grad = requires_grad
        return self

    def backward(self, gradient=None, *, retain_graph=False, create_graph=False):
        if create_graph: raise BrowserUnsupported('Higher-order autograd is not supported')
        if gradient is None and self.numel() != 1: raise RuntimeError('Non-scalar outputs require a gradient')
        seed = np.ones_like(self._data) if gradient is None else wrap(gradient)._data
        order, gradients = backprop(self, seed)
        for node in order:
            if node.requires_grad and not node._parents and id(node) in gradients:
                grad = gradients[id(node)]
                node.grad = Tensor(grad.copy() if node.grad is None else node.grad._data + grad)

    def __getitem__(self, index):
        def convert(key):
            if isinstance(key, Tensor): return key._data
            if isinstance(key, tuple): return tuple(convert(k) for k in key)
            return key
        index = convert(index)
        def vjp(g):
            grad = np.zeros_like(self._data)
            np.add.at(grad, index, g)
            return grad
        return result(self._data[index], (self, vjp))

    def __setitem__(self, index, value):
        if self.requires_grad and _grad_enabled:
            raise BrowserUnsupported('In-place edits of differentiable tensors require the server')
        if isinstance(index, Tensor): index = index._data
        self._data[index] = value._data if isinstance(value, Tensor) else value

    def _binary(self, other, op, left, right):
        if not isinstance(other, (Tensor, np.ndarray)) and self.dtype.kind == 'f':
            other = Tensor(other, dtype=self.dtype)
        other = wrap(other)
        a, b = self._data, other._data
        floating = [value.dtype for value in (a,b) if value.dtype.kind == 'f']
        if floating:
            dtype = np.result_type(*floating)
            if a.ndim and not b.ndim and a.dtype.kind == 'f': dtype = a.dtype
            if b.ndim and not a.ndim and b.dtype.kind == 'f': dtype = b.dtype
            a, b = a.astype(dtype,copy=False), b.astype(dtype,copy=False)
        return result(op(a, b), (self, lambda g: unbroadcast(left(g, a, b), a.shape)),
                      (other, lambda g: unbroadcast(right(g, a, b), b.shape)))
    def __add__(self, other): return self._binary(other, np.add, lambda g,a,b:g, lambda g,a,b:g)
    __radd__ = __add__
    def __sub__(self, other): return self._binary(other, np.subtract, lambda g,a,b:g, lambda g,a,b:-g)
    def __rsub__(self, other): return -self + other
    def __mul__(self, other): return self._binary(other, np.multiply, lambda g,a,b:g*b, lambda g,a,b:g*a)
    __rmul__ = __mul__
    def __truediv__(self, other): return self._binary(other, np.divide, lambda g,a,b:g/b, lambda g,a,b:-g*a/(b*b))
    def __rtruediv__(self, other): return self._binary(other, lambda a,b: b/a, lambda g,a,b: -g*b/(a*a), lambda g,a,b: g/a)
    def __neg__(self): return result(-self._data, (self, lambda g: -g))
    def __pow__(self, exponent):
        if not np.isscalar(exponent): raise BrowserUnsupported('Tensor exponents require the server')
        return result(self._data ** exponent, (self, lambda g: g * exponent * self._data ** (exponent - 1)))
    def pow(self, exponent): return self ** exponent
    def square(self): return self * self
    def sqrt(self): return self ** .5
    def abs(self): return result(np.abs(self._data), (self, lambda g: g * np.sign(self._data)))
    def relu(self):
        return result(np.maximum(self._data, 0), (self, lambda g: np.where(self._data <= 0, 0, g)))
    def exp(self):
        data = np.exp(self._data)
        return result(data, (self, lambda g: g * data))
    def log(self): return result(np.log(self._data), (self, lambda g: g / self._data))

    def __matmul__(self, other):
        other = wrap(other)
        a, b = self._data, other._data
        if a.dtype != b.dtype: raise RuntimeError('matmul operands must have the same dtype')
        # Promote vectors to matrices before calculating batched VJPs.
        aa, bb = a[None, :] if a.ndim == 1 else a, b[:, None] if b.ndim == 1 else b
        def promote(g):
            if b.ndim == 1: g = np.expand_dims(g, -1)
            if a.ndim == 1: g = np.expand_dims(g, -2)
            return g
        def left(g):
            grad = unbroadcast(promote(g) @ np.swapaxes(bb, -1, -2), aa.shape)
            return grad.reshape(a.shape)
        def right(g):
            grad = unbroadcast(np.swapaxes(aa, -1, -2) @ promote(g), bb.shape)
            return grad.reshape(b.shape)
        return result(a @ b, (self, left), (other, right))
    def __rmatmul__(self, other): return wrap(other) @ self

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)): shape = tuple(shape[0])
        return result(self._data.reshape(shape), (self, lambda g: g.reshape(self.shape)))
    view = reshape
    def flatten(self, start_dim=0, end_dim=-1):
        if not self.ndim: return self.reshape(1)
        end_dim %= self.ndim
        return self.reshape(*self.shape[:start_dim], -1, *self.shape[end_dim+1:])
    def unsqueeze(self, dim): return result(np.expand_dims(self._data, dim), (self, lambda g: g.reshape(self.shape)))
    def squeeze(self, dim=None):
        if dim is not None and self.shape[dim] != 1: return self
        return result(np.squeeze(self._data, axis=dim), (self, lambda g: g.reshape(self.shape)))
    def permute(self, *dims):
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)): dims = tuple(dims[0])
        dims = tuple(d % self.ndim for d in dims)
        return result(self._data.transpose(dims), (self, lambda g: g.transpose(np.argsort(dims))))
    def transpose(self, dim0, dim1):
        dims = list(range(self.ndim)); dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
        return self.permute(*dims)
    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)): shape = tuple(shape[0])
        leading = (1,) * (len(shape) - self.ndim) + self.shape
        target = tuple(old if new == -1 else new for old, new in zip(leading, shape))
        return result(np.broadcast_to(self._data, target), (self, lambda g: unbroadcast(g, self.shape)))
    def expand_as(self, other): return self.expand(other.shape)
    def repeat(self, *repeats):
        if len(repeats) == 1 and isinstance(repeats[0], (tuple, list)): repeats = tuple(repeats[0])
        shape = (1,) * (len(repeats) - self.ndim) + self.shape
        interleaved = tuple(v for pair in zip(repeats, shape) for v in pair)
        return result(np.tile(self._data, repeats), (self, lambda g: g.reshape(interleaved).sum(axis=tuple(range(0, len(interleaved), 2))).reshape(self.shape)))

    def sum(self, dim=None, keepdim=False, *, dtype=None):
        dims = tuple(range(self.ndim)) if dim is None else tuple(d % self.ndim for d in (dim if isinstance(dim, (list, tuple)) else (dim,)))
        def vjp(g):
            if not keepdim:
                for axis in sorted(dims): g = np.expand_dims(g, axis)
            return np.broadcast_to(g, self.shape)
        if dtype is None and self.dtype.kind in 'biu': dtype = np.int64
        return result(self._data.sum(axis=dims, keepdims=keepdim, dtype=dtype), (self, vjp))
    def mean(self, dim=None, keepdim=False):
        dims = tuple(range(self.ndim)) if dim is None else dim if isinstance(dim, (list, tuple)) else (dim,)
        return self.sum(dim, keepdim) / int(np.prod([self.shape[d] for d in dims]))
    def any(self, dim=None, keepdim=False): return Tensor(self._data.any(axis=dim, keepdims=keepdim))
    def all(self, dim=None, keepdim=False): return Tensor(self._data.all(axis=dim, keepdims=keepdim))
    def _extreme(self, dim, keepdim, maximum):
        op, argop = (np.max, np.argmax) if maximum else (np.min, np.argmin)
        data = op(self._data, axis=dim, keepdims=keepdim)
        if dim is None:
            mask = self._data == data
            return result(data, (self, lambda g: g * mask / mask.sum()))
        indices = argop(self._data, axis=dim)
        def vjp(g):
            grad = np.zeros_like(self._data)
            np.put_along_axis(grad, np.expand_dims(indices, dim), g if keepdim else np.expand_dims(g, dim), axis=dim)
            return grad
        pair = namedtuple('Reduction', ['values', 'indices'])
        return pair(result(data, (self, vjp)), Tensor(np.expand_dims(indices, dim) if keepdim else indices))
    def max(self, dim=None, keepdim=False): return self._extreme(dim, keepdim, True)
    def min(self, dim=None, keepdim=False): return self._extreme(dim, keepdim, False)
    def argmax(self, dim=None, keepdim=False):
        value = np.argmax(self._data, axis=dim)
        return Tensor(np.expand_dims(value, dim) if keepdim and dim is not None else value)
    def cumsum(self, dim):
        return result(self._data.cumsum(axis=dim), (self, lambda g: np.flip(np.flip(g, axis=dim).cumsum(axis=dim), axis=dim)))
    def softmax(self, dim, dtype=None):
        x = self.to(dtype=dtype) if dtype is not None else self
        shifted = x._data - x._data.max(axis=dim, keepdims=True)
        exps = np.exp(shifted)
        probabilities = exps / exps.sum(axis=dim, keepdims=True)
        return result(probabilities, (x, lambda g: probabilities * (g - (g * probabilities).sum(axis=dim, keepdims=True))))
    def masked_fill(self, mask, value):
        if not isinstance(value, Tensor): value = Tensor(value, dtype=self.dtype)
        return where(mask, value, self)
    def tril(self, diagonal=0): return self * Tensor(np.tril(np.ones(self.shape[-2:], dtype=self.dtype), diagonal))
    def triu(self, diagonal=0): return self * Tensor(np.triu(np.ones(self.shape[-2:], dtype=self.dtype), diagonal))
    def clamp(self, min=None, max=None):
        if min is None and max is None: raise ValueError('clamp needs at least one bound')
        if isinstance(min, Tensor) or isinstance(max, Tensor): raise BrowserUnsupported('Tensor clamp bounds require the server')
        mask = np.ones(self.shape, dtype=bool)
        if min is not None: mask &= self._data >= min
        if max is not None: mask &= self._data <= max
        return result(np.clip(self._data, min, max), (self, lambda g: g * mask))
    def clamp_min(self, min): return self.clamp(min=min)
    def clamp_max(self, max): return self.clamp(max=max)
    def sin(self): return result(np.sin(self._data), (self, lambda g: g * np.cos(self._data)))
    def cos(self): return result(np.cos(self._data), (self, lambda g: -g * np.sin(self._data)))
    def acos(self): return result(np.arccos(self._data), (self, lambda g: -g / np.sqrt(1 - self._data ** 2)))
    def tanh(self):
        value = np.tanh(self._data)
        return result(value, (self, lambda g: g * (1 - value ** 2)))
    def rsqrt(self): return self ** -.5
    def sigmoid(self):
        exp = np.exp(-np.abs(self._data))
        value = np.where(self._data >= 0, 1 / (1 + exp), exp / (1 + exp))
        return result(value, (self, lambda g: g * value * (1 - value)))
    def logsigmoid(self):
        zero = np.zeros((), dtype=self.dtype)
        return result(-np.logaddexp(zero, -self._data), (self, lambda g: g * np.exp(-np.logaddexp(zero, self._data))))
    def log_softmax(self, dim, dtype=None):
        x = self.to(dtype=dtype) if dtype is not None else self
        shifted = x._data - x._data.max(axis=dim, keepdims=True)
        value = shifted - np.log(np.exp(shifted).sum(axis=dim, keepdims=True))
        return result(value, (x, lambda g: g - np.exp(value) * g.sum(axis=dim, keepdims=True)))
    def gather(self, dim, index):
        dim %= self.ndim
        index = wrap(index)._data
        if index.ndim != self.ndim or index.dtype.kind not in 'iu': raise ValueError('gather needs integer indices of the same rank')
        if np.any(index < 0) or np.any(index >= self.shape[dim]): raise IndexError('gather index out of bounds')
        if any(index.shape[d] > self.shape[d] for d in range(self.ndim) if d != dim): raise ValueError('gather dimensions do not match')
        slices = tuple(slice(None) if d == dim else slice(0, index.shape[d]) for d in range(self.ndim))
        def vjp(g):
            coordinates = list(np.indices(index.shape)); coordinates[dim] = index
            grad = np.zeros_like(self._data); np.add.at(grad, tuple(coordinates), g)
            return grad
        return result(np.take_along_axis(self._data[slices], index, axis=dim), (self, vjp))
    def var(self, dim=None, unbiased=True, keepdim=False, *, correction=None):
        dims = tuple(range(self.ndim)) if dim is None else tuple(d % self.ndim for d in (dim if isinstance(dim, (list, tuple)) else (dim,)))
        count = int(np.prod([self.shape[d] for d in dims]))
        correction = int(unbiased) if correction is None else correction
        centered = self - self.mean(dim, keepdim=True)
        return centered.square().sum(dim, keepdim) / (count - correction)
    def std(self, dim=None, unbiased=True, keepdim=False, *, correction=None):
        dims = tuple(range(self.ndim)) if dim is None else tuple(d % self.ndim for d in (dim if isinstance(dim, (list, tuple)) else (dim,)))
        correction = int(unbiased) if correction is None else correction
        count = int(np.prod([self.shape[d] for d in dims]))
        value = np.std(self._data, axis=dims, ddof=correction, keepdims=True)
        centered = self._data - self._data.mean(axis=dims, keepdims=True)
        def vjp(g):
            if not keepdim:
                for axis in sorted(dims): g = np.expand_dims(g, axis)
            derivative = np.divide(centered, value * (count - correction), out=np.zeros_like(centered), where=value != 0)
            return g * derivative
        return result(value if keepdim else value.squeeze(axis=dims), (self, vjp))

    def _compare(self, other, op): return Tensor(op(self._data, wrap(other)._data))
    def __eq__(self, other): return self._compare(other, np.equal)
    def __ne__(self, other): return self._compare(other, np.not_equal)
    def __lt__(self, other): return self._compare(other, np.less)
    def __le__(self, other): return self._compare(other, np.less_equal)
    def __gt__(self, other): return self._compare(other, np.greater)
    def __ge__(self, other): return self._compare(other, np.greater_equal)
    def __invert__(self): return Tensor(~self._data)
    def __and__(self, other): return Tensor(self._data & wrap(other)._data)
    def __or__(self, other): return Tensor(self._data | wrap(other)._data)


def where(condition, x, y):
    mask, a, b = wrap(condition)._data, wrap(x), wrap(y)
    return result(np.where(mask, a._data, b._data),
                  (a, lambda g: unbroadcast(np.where(mask, g, 0), a.shape)),
                  (b, lambda g: unbroadcast(np.where(mask, 0, g), b.shape)))


def tensor(data, dtype=None, *, device=None, requires_grad=False):
    check_device(device)
    if isinstance(data, Tensor): data = data._data.copy()
    array = np.asarray(data)
    if dtype is None and not isinstance(data, (np.ndarray, np.generic)):
        dtype = np.float32 if array.dtype.kind == 'f' else np.int64 if array.dtype.kind in 'iu' else None
    return Tensor(array, dtype=dtype, requires_grad=requires_grad)


def check_device(device):
    if device not in (None, 'cpu', 'cpu:0'): raise BrowserUnsupported('Only CPU/WASM tensors are supported')


class Generator:
    def __init__(self, device=None): check_device(device); self.rng = np.random.RandomState(42)
    def manual_seed(self, seed): self.rng = np.random.RandomState(seed); return self


def create(kind, *shape, dtype=np.float32, device=None, requires_grad=False, generator=None):
    check_device(device)
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)): shape = tuple(shape[0])
    rng = generator.rng if generator else np.random
    value = rng.standard_normal(shape) if kind == 'randn' else uniform_samples(rng, shape, dtype) if kind == 'rand' else getattr(np, kind)(shape, dtype=dtype)
    return Tensor(value, dtype=dtype, requires_grad=requires_grad)


def einsum(equation, a, b):
    a, b = wrap(a), wrap(b)
    inputs, output = equation.replace(' ', '').split('->')
    left, right = inputs.split(',')
    if '.' in equation or len(set(left)) != len(left) or len(set(right)) != len(right):
        raise BrowserUnsupported('This einsum pattern requires the server')
    return result(np.einsum(equation, a._data, b._data),
                  (a, lambda g: unbroadcast(np.einsum(f'{output},{right}->{left}', g, b._data), a.shape)),
                  (b, lambda g: unbroadcast(np.einsum(f'{output},{left}->{right}', g, a._data), b.shape)))


def autograd_grad(output, inputs, grad_outputs=None, *, retain_graph=False, create_graph=False, allow_unused=False):
    if create_graph: raise BrowserUnsupported('Higher-order autograd requires the server')
    if isinstance(output, (tuple, list)):
        if len(output) != 1: raise BrowserUnsupported('Multiple-output autograd requires the server')
        output = output[0]
    if grad_outputs is None and output.numel() != 1: raise RuntimeError('Non-scalar outputs require grad_outputs')
    _, gradients = backprop(output, np.ones_like(output._data) if grad_outputs is None else wrap(grad_outputs)._data)
    values = []
    for node in ([inputs] if isinstance(inputs, Tensor) else inputs):
        if id(node) not in gradients:
            if not allow_unused: raise RuntimeError('A differentiated tensor was not used in the graph')
            values.append(None)
        else: values.append(Tensor(gradients[id(node)]))
    return tuple(values)


class no_grad(contextlib.ContextDecorator):
    def __enter__(self):
        global _grad_enabled
        self.before = _grad_enabled; _grad_enabled = False
        return self
    def __exit__(self, *args):
        global _grad_enabled
        _grad_enabled = self.before


class Adam:
    def __getattr__(self, name):
        if name.startswith('_'): raise AttributeError(name)
        return unsupported(f'Adam.{name}')

    def __init__(self, params, lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0):
        if weight_decay: raise BrowserUnsupported('Adam weight decay requires the server')
        self.params, self.lr, self.betas, self.eps = list(params), lr, betas, eps
        self.steps = 0
        self.m = [np.zeros_like(p._data) for p in self.params]
        self.v = [np.zeros_like(p._data) for p in self.params]
    def zero_grad(self, set_to_none=True):
        for p in self.params: p.grad = None if set_to_none else Tensor(np.zeros_like(p._data))
    def step(self):
        self.steps += 1
        b1, b2 = self.betas
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            g = p.grad._data
            self.m[i] *= b1; self.m[i] += (1 - b1) * g
            self.v[i] *= b2; self.v[i] += (1 - b2) * g * g
            p._data[...] -= self.lr * (self.m[i] / (1 - b1 ** self.steps)) / (np.sqrt(self.v[i] / (1 - b2 ** self.steps)) + self.eps)


class AdamW:
    """Independent, stateful NumPy oracle for the functional AdamW exercise."""
    def __getattr__(self, name):
        if name.startswith('_'): raise AttributeError(name)
        return unsupported(f'AdamW.{name}')

    def __init__(self, params, lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=.01, **options):
        if options: raise BrowserUnsupported('Advanced AdamW options require the server')
        params = list(params)
        if not params: raise ValueError('AdamW needs parameters')
        if any(not isinstance(p, Tensor) for p in params):
            raise BrowserUnsupported('AdamW parameter groups require the server')
        if lr < 0 or eps < 0 or weight_decay < 0 or len(betas) != 2 or any(not 0 <= b < 1 for b in betas):
            raise ValueError('Invalid AdamW options')
        self.param_groups = [dict(params=params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)]
        self.state = {}

    def zero_grad(self, set_to_none=True):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad = None if set_to_none else Tensor(np.zeros_like(p._data))

    def step(self, closure=None):
        if closure is not None: raise BrowserUnsupported('AdamW closures require the server')
        for group in self.param_groups:
            b1, b2 = group['betas']
            for p in group['params']:
                if p.grad is None: continue
                if p.grad.shape != p.shape or p.grad.dtype != p.dtype:
                    raise RuntimeError('AdamW gradient shape/dtype must match the parameter')
                if p not in self.state:
                    self.state[p] = dict(step=Tensor(np.array(0., dtype=np.float32)),
                                         exp_avg=Tensor(np.zeros_like(p._data)), exp_avg_sq=Tensor(np.zeros_like(p._data)))
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


def concatenate(tensors, dim=0, *, stack=False):
    tensors = [wrap(value) for value in tensors]
    if not tensors: raise RuntimeError('Expected a non-empty tensor sequence')
    rank = tensors[0].ndim + int(stack)
    if not -rank <= dim < rank: raise IndexError('Tensor dimension out of range')
    dim %= rank
    floating = [value.dtype for value in tensors if value.dtype.kind == 'f']
    dtype = np.result_type(*(floating or [value.dtype for value in tensors]))
    values = [value._data.astype(dtype, copy=False) for value in tensors]
    data = np.stack(values, axis=dim) if stack else np.concatenate(values, axis=dim)
    edges, offset = [], 0
    for value in tensors:
        start, end = offset, offset + (1 if stack else value.shape[dim])
        def vjp(g, start=start, end=end):
            index = [slice(None)] * rank
            index[dim] = start if stack else slice(start, end)
            return g[tuple(index)]
        edges.append((value, vjp))
        offset = end
    return result(data, *edges)


def scaled_dot_product_attention_oracle(query, key, value, attn_mask=None, dropout_p=0.,
                                        is_causal=False, *, scale=None, enable_gqa=False):
    """Raw NumPy SDPA with its own analytical VJP, independent of learner ops."""
    if dropout_p or enable_gqa or (is_causal and attn_mask is not None):
        raise BrowserUnsupported('Dropout, grouped-query or combined causal/masked SDPA requires the server')
    q, k, v = query._data, key._data, value._data
    if q.ndim < 2 or k.ndim < 2 or v.ndim < 2 or q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2]:
        raise RuntimeError('Invalid SDPA shapes')
    if q.dtype != k.dtype or q.dtype != v.dtype or q.dtype.kind != 'f':
        raise RuntimeError('SDPA expects floating tensors with matching dtypes')
    factor = 1 / np.sqrt(q.shape[-1]) if scale is None else float(scale)
    scores = (q @ np.swapaxes(k, -1, -2)) * factor
    if is_causal:
        scores = np.where(np.arange(k.shape[-2])[None, :] <= np.arange(q.shape[-2])[:, None], scores, -np.inf)
    if attn_mask is not None:
        mask = wrap(attn_mask)._data
        if mask.dtype.kind == 'b': scores = np.where(mask, scores, -np.inf)
        elif mask.dtype == q.dtype: scores = scores + mask
        else: raise RuntimeError('SDPA mask must be boolean or match the query dtype')
    if scores.shape[-1]:
        maximum = scores.max(axis=-1, keepdims=True)
        exponentials = np.exp(scores - np.where(maximum == -np.inf, 0, maximum))
        denominator = exponentials.sum(axis=-1, keepdims=True)
        probabilities = np.divide(exponentials, denominator, out=np.zeros_like(exponentials), where=denominator != 0)
    else:
        probabilities = np.zeros_like(scores)

    def logits_vjp(g):
        incoming = g @ np.swapaxes(v, -1, -2)
        return probabilities * (incoming - (incoming * probabilities).sum(axis=-1, keepdims=True))

    edges = [(query, lambda g: unbroadcast((logits_vjp(g) @ k) * factor, q.shape)),
             (key, lambda g: unbroadcast((np.swapaxes(logits_vjp(g), -1, -2) @ q) * factor, k.shape)),
             (value, lambda g: unbroadcast(np.swapaxes(probabilities, -1, -2) @ g, v.shape))]
    if isinstance(attn_mask, Tensor) and attn_mask.dtype.kind == 'f':
        edges.append((attn_mask, lambda g: unbroadcast(logits_vjp(g), attn_mask.shape)))
    return result(probabilities @ v, *edges)


def layer_norm_oracle(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    """Separate analytical LayerNorm VJP for the library-oracle test path."""
    x = wrap(x)
    shape = tuple(normalized_shape)
    if not shape or tuple(x.shape[-len(shape):]) != shape: raise ValueError('normalized_shape does not match input')
    axes = tuple(range(x.ndim-len(shape), x.ndim))
    centered = x._data - x._data.mean(axis=axes, keepdims=True)
    inverse = 1 / np.sqrt(np.mean(centered ** 2, axis=axes, keepdims=True) + eps)
    normalized = centered * inverse
    scale = np.ones(shape, dtype=x.dtype) if weight is None else wrap(weight)._data
    value = normalized * scale
    if bias is not None: value = value + wrap(bias)._data
    def dx(g):
        scaled = g * scale
        return inverse * (scaled - scaled.mean(axis=axes, keepdims=True) - normalized * (scaled * normalized).mean(axis=axes, keepdims=True))
    edges = [(x, dx)]
    if weight is not None: edges.append((weight, lambda g: unbroadcast(g * normalized, weight.shape)))
    if bias is not None: edges.append((bias, lambda g: unbroadcast(g, bias.shape)))
    return result(value, *edges)


def randint(low, high, size, *, generator=None, dtype=np.int64, device=None):
    check_device(device)
    rng = generator.rng if generator else np.random
    if not 0 < high-low <= 2**32: raise BrowserUnsupported('This randint range requires the server')
    bits = rng.randint(0, 2**32, size=size, dtype=np.uint32).astype(np.uint64)
    return Tensor((bits % (high-low)).astype(dtype) + low)


def full(shape, fill_value, *, dtype=None, device=None, requires_grad=False):
    check_device(device)
    if dtype is None: dtype = np.bool_ if isinstance(fill_value, bool) else np.float32 if isinstance(fill_value, float) else np.int64
    return Tensor(np.full(shape, fill_value, dtype=dtype), requires_grad=requires_grad)


class Device(str):
    def __new__(cls, name='cpu'):
        check_device(name)
        return str.__new__(cls, name)
    type = property(lambda self: 'cpu')
    index = property(lambda self: 0 if self.endswith(':0') else None)


def uniform_samples(rng, shape, dtype):
    # CPU MT19937 uniforms use the low mantissa bits, unlike NumPy's
    # random_sample conversion. Keep preference-pair sampling reproducible.
    if np.dtype(dtype) == np.dtype('float64'):
        bits = rng.randint(0,2**32,size=tuple(shape)+(2,),dtype=np.uint32).astype(np.uint64)
        raw = ((bits[...,0] << 32) | bits[...,1]) & np.uint64((1 << 53)-1)
        return raw.astype(np.float64) / (1 << 53)
    bits = rng.randint(0,2**32,size=shape,dtype=np.uint32)
    return (bits & ((1 << 24)-1)).astype(np.float32) / (1 << 24)


def pdist(x, p=2):
    if p != 2 or x.ndim != 2: raise BrowserUnsupported('Only 2D Euclidean pdist is supported')
    i,j = np.triu_indices(x.shape[0],k=1)
    difference = x._data[i] - x._data[j]
    lengths = np.linalg.norm(difference,axis=1)
    def vjp(g):
        grad = np.zeros_like(x._data)
        direction = np.divide(difference,lengths[:,None],out=np.zeros_like(difference),where=lengths[:,None]!=0)
        np.add.at(grad,i,g[:,None]*direction);np.add.at(grad,j,-g[:,None]*direction)
        return grad
    return result(lengths,(x,vjp))


def install():
    torch = types.ModuleType('torch')
    torch.__path__ = []
    torch.Tensor, torch.Generator, torch.no_grad = Tensor, Generator, no_grad
    torch.float64, torch.float32 = np.dtype('float64'), np.dtype('float32')
    torch.float, torch.double = torch.float32, torch.float64
    torch.long, torch.int64, torch.bool = np.dtype('int64'), np.dtype('int64'), np.dtype('bool')
    torch.inf, torch.pi = np.inf, np.pi
    torch.tensor, torch.where, torch.einsum = tensor, where, einsum
    torch.as_tensor = lambda x: x if isinstance(x, Tensor) else tensor(x)
    torch.from_numpy = lambda x: Tensor(x)
    torch.full, torch.randint, torch.device, torch.finfo = full, randint, Device, np.finfo
    torch.eye = lambda n,m=None,*,dtype=np.float32,device=None,requires_grad=False: (check_device(device), Tensor(np.eye(n, m, dtype=dtype), requires_grad=requires_grad))[1]
    torch.equal = lambda a,b: bool(np.array_equal(wrap(a)._data, wrap(b)._data))
    torch.cat = concatenate
    torch.stack = lambda tensors,dim=0: concatenate(tensors,dim,stack=True)
    torch.linspace = lambda start,end,steps,dtype=np.float32,device=None: (check_device(device), Tensor(np.linspace(start,end,steps,dtype=dtype)))[1]
    torch.argsort = lambda x,dim=-1,descending=False: Tensor(np.argsort(x._data,axis=dim) if not descending else np.flip(np.argsort(x._data,axis=dim),axis=dim),dtype=np.int64)
    torch.manual_seed = np.random.seed
    for name in ('randn', 'rand', 'zeros', 'ones', 'empty'):
        setattr(torch, name, lambda *shape, _kind=name, **kw: create(_kind, *shape, **kw))
    for name in ('randn', 'zeros', 'ones', 'empty'):
        setattr(torch, name + '_like', lambda x, _kind=name, **kw: create(_kind, *x.shape, **({'dtype': x.dtype} | kw)))
    torch.arange = lambda *args, dtype=None, device=None: (check_device(device), Tensor(np.arange(*args, dtype=dtype or (np.float32 if any(isinstance(x, float) for x in args) else np.int64))))[1]
    torch.ones_like = lambda x, **kw: create('ones', *x.shape, **({'dtype': x.dtype} | kw))
    torch.full_like = lambda x, value: Tensor(np.full_like(x._data, value))
    torch.unravel_index = lambda index, shape: tuple(Tensor(i) for i in np.unravel_index(int(index), shape))
    torch.broadcast_to = lambda x, shape: x.expand(shape)
    torch.matmul = lambda a,b: wrap(a) @ b
    torch.pdist = pdist
    for name in ('sum', 'mean', 'softmax', 'exp', 'log', 'sqrt', 'abs', 'relu', 'transpose', 'reshape', 'flatten', 'max', 'min', 'clamp', 'clamp_min', 'clamp_max', 'sin', 'cos', 'acos', 'tanh', 'rsqrt', 'sigmoid', 'log_softmax', 'var', 'std'):
        setattr(torch, name, lambda x, *args, _name=name, **kw: getattr(wrap(x), _name)(*args, **kw))
    for name in ('isfinite', 'isnan', 'count_nonzero'):
        setattr(torch, name, lambda x, _name=name: Tensor(getattr(np, _name)(wrap(x)._data)))
    torch.isclose = lambda a,b,**kw: Tensor(np.isclose(wrap(a)._data, wrap(b)._data, **kw))
    torch.allclose = lambda a,b,**kw: bool(torch.isclose(a,b,**kw).all())
    torch.nan_to_num = lambda x,**kw: Tensor(np.nan_to_num(x._data, **kw))
    modules = {}
    for name in ('nn', 'nn.functional', 'optim', 'autograd', 'linalg'):
        module = types.ModuleType('torch.' + name)
        module.__getattr__ = lambda key, _name=name: unsupported('torch.' + _name + '.' + key)
        modules[name] = module
        sys.modules['torch.' + name] = module
    torch.nn, torch.optim, torch.autograd, torch.linalg = (modules[k] for k in ('nn', 'optim', 'autograd', 'linalg'))
    torch.nn.functional = modules['nn.functional']
    torch.nn.functional.linear = lambda x,w,bias=None: x @ w.T if bias is None else x @ w.T + bias
    torch.nn.functional.softmax = lambda x,dim,dtype=None: x.softmax(dim, dtype)
    torch.nn.functional.log_softmax = lambda x,dim,dtype=None: x.log_softmax(dim,dtype)
    torch.nn.functional.logsigmoid = lambda x: x.logsigmoid()
    torch.nn.functional.layer_norm = layer_norm_oracle
    torch.nn.functional.scaled_dot_product_attention = scaled_dot_product_attention_oracle
    def relu(x, inplace=False):
        if inplace: raise BrowserUnsupported('In-place ReLU requires the server')
        return x.relu()
    torch.nn.functional.relu = relu
    torch.optim.Adam, torch.autograd.grad = Adam, autograd_grad
    torch.optim.AdamW = AdamW
    torch.linalg.matrix_rank = lambda x: Tensor(np.linalg.matrix_rank(x._data), dtype=np.int64)
    torch.__getattr__ = unsupported
    return torch

"""Bounded NumPy reverse-mode autodiff backend for validated teaching modules.

CPU/WASM only. Gradients are analytical vector-Jacobian products, not numerical
approximations. This is a teaching-subset adapter, not a native PyTorch runtime.
Unsupported APIs raise BrowserUnsupported and retain server judging.
"""
import contextlib
import copy
import math
import types
import sys
import weakref
from collections import namedtuple
from statistics import NormalDist
import numpy as np


class BrowserUnsupported(NotImplementedError):
    pass


_grad_enabled = True
_storages = {}


class StorageVersion:
    def __init__(self): self.version = 0


def storage_version(data):
    """Views, detach and dtype-preserving casts share their NumPy allocation's counter."""
    owner = data
    while isinstance(owner.base, np.ndarray): owner = owner.base
    key = id(owner)
    previous = _storages.get(key)
    if previous is not None and previous[0]() is owner: return previous[1]
    counter = StorageVersion()
    def release(reference):
        if _storages.get(key, (None,))[0] is reference: _storages.pop(key, None)
    _storages[key] = (weakref.ref(owner, release), counter)
    return counter


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


def inherited_options(dtype, options):
    options=dict(options)
    if options.get('dtype') is None: options['dtype']=dtype
    return options


def reduction_axes(dim, ndim):
    if dim is None: return tuple(range(ndim))
    dims = tuple(dim) if isinstance(dim, (list, tuple)) else (dim,)
    if not dims: raise BrowserUnsupported('Empty reduction dimension lists require the server')
    if any(not isinstance(d, (int, np.integer)) or not -max(ndim, 1) <= d < max(ndim, 1) for d in dims):
        raise IndexError('Reduction dimension out of range')
    axes = tuple(d % ndim for d in dims) if ndim else ()
    if len(set(axes)) != len(axes): raise RuntimeError('Duplicate reduction dimensions')
    return axes


def result(data, *edges, saved=()):
    value = Tensor(data)
    if value.dtype.kind == 'c' and _grad_enabled and any(node.requires_grad for node, _ in edges):
        raise BrowserUnsupported('Complex autograd requires the server')
    value._parents = tuple((node, vjp) for node, vjp in edges if node.requires_grad and _grad_enabled)
    value.requires_grad = bool(value._parents)
    if value.requires_grad:
        # Conservatively guard every operand, including frozen operands and output
        # arrays captured by VJP closures. Unsupported graph reuse falls back rather
        # than returning derivatives calculated from mutated storage.
        counters = {node._storage for node in (value, *(node for node, _ in edges), *saved)}
        value._saved_versions = tuple((counter, counter.version) for counter in counters)
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
    for node in order:
        if any(counter.version != version for counter, version in node._saved_versions):
            raise BrowserUnsupported('A tensor saved for backward was modified in-place; this graph requires the server')
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
        self._storage = storage_version(self._data)
        self._saved_versions = ()
        self.requires_grad = bool(requires_grad)
        if requires_grad and self.dtype.kind == 'c': raise BrowserUnsupported('Complex autograd requires the server')
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
    def _real_only(self):
        if self.dtype.kind == 'c': raise BrowserUnsupported('This complex operation requires the server')
    def __array__(self, dtype=None, copy=None): return np.array(self._data, dtype=dtype, copy=copy) if copy is not None else np.asarray(self._data, dtype=dtype)
    def __len__(self): return len(self._data)
    def __float__(self): return float(self._data)
    def __int__(self): return int(self._data)
    def __index__(self):
        if self.numel() != 1 or self.dtype.kind not in 'biu':
            raise TypeError('Only integer tensors of a single element can be used as an index')
        return int(self._data.item())
    def __bool__(self): return bool(self._data)
    def __repr__(self): return f'Tensor({self._data!r}, requires_grad={self.requires_grad})'
    def item(self): return self._data.item()
    def tolist(self): return self._data.tolist()
    def numpy(self): return self._data
    def numel(self): return self._data.size
    def data_ptr(self): return int(self._data.__array_interface__['data'][0]) if self.numel() else 0
    def dim(self): return self.ndim
    def is_floating_point(self): return self.dtype.kind == 'f'
    def count_nonzero(self, dim=None): return count_nonzero(self, dim)
    def size(self, dim=None): return self.shape if dim is None else self.shape[dim]
    def detach(self): return Tensor(self._data)
    def clone(self): return result(self._data.copy(), (self, lambda g: g))
    def cpu(self): return self
    def contiguous(self): return self.clone()
    def is_contiguous(self): return bool(self._data.flags.c_contiguous)
    def double(self): return self.to(dtype=np.float64)
    def float(self): return self.to(dtype=np.float32)
    def long(self): return self.to(dtype=np.int64)
    def int(self): return self.to(dtype=np.int32)
    def bool(self): return self.to(dtype=np.bool_)
    def new_zeros(self, *shape, **kw): return create('zeros', *shape, **inherited_options(self.dtype,kw))
    def new_ones(self, *shape, **kw): return create('ones', *shape, **inherited_options(self.dtype,kw))
    def new_empty(self, *shape, **kw): return create('empty', *shape, **inherited_options(self.dtype,kw))
    def new_full(self, size, fill_value, **kw): return full(size, fill_value, **inherited_options(self.dtype,kw))
    def new_tensor(self, data, *, dtype=None, device=None, requires_grad=False):
        check_device(device)
        return Tensor(np.array(data._data if isinstance(data, Tensor) else data, dtype=dtype or self.dtype, copy=True), requires_grad=requires_grad)
    def zero_(self): self[...] = 0; return self
    def type_as(self, other): return self.to(other)

    def to(self, other=None, *, dtype=None, device=None):
        if isinstance(other, Tensor): dtype = other.dtype
        elif isinstance(other, str) and other.startswith('cpu'): device = other
        elif other is not None: dtype = other
        if device not in (None, 'cpu', 'cpu:0'):
            raise BrowserUnsupported('Only CPU/WASM tensors are supported')
        target = np.dtype(dtype or self.dtype)
        if self.dtype.kind == 'c' and target.kind != 'c':
            raise BrowserUnsupported('Casting complex tensors to real values requires the server')
        data = self._data.astype(target, copy=False)
        return result(data, (self, lambda g: g.astype(self.dtype))) if data.dtype.kind in 'fc' else Tensor(data)

    def requires_grad_(self, requires_grad=True):
        if requires_grad and self.dtype.kind == 'c': raise BrowserUnsupported('Complex autograd requires the server')
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
        saved = []
        def convert(key):
            if isinstance(key, Tensor): saved.append(key); return key._data
            if isinstance(key, tuple): return tuple(convert(k) for k in key)
            return key
        index = convert(index)
        def vjp(g):
            grad = np.zeros_like(self._data)
            np.add.at(grad, index, g)
            return grad
        return result(self._data[index], (self, vjp), saved=saved)

    def __setitem__(self, index, value):
        if _grad_enabled and (self.requires_grad or isinstance(value, Tensor) and value.requires_grad):
            raise BrowserUnsupported('In-place edits of differentiable tensors require the server')
        if self.requires_grad and self._parents:
            raise BrowserUnsupported('In-place updates of non-leaf tensors require the server')
        def convert(key):
            if isinstance(key, Tensor): return key._data
            if isinstance(key, tuple): return tuple(convert(k) for k in key)
            return key
        index = convert(index)
        self._data[index] = value._data if isinstance(value, Tensor) else value
        self._storage.version += 1

    def fill_(self, value):
        if isinstance(value, Tensor) and value.ndim:
            raise RuntimeError('fill_ requires a scalar value')
        self[...] = value
        return self

    def __deepcopy__(self, memo):
        if self._parents:
            raise BrowserUnsupported('Copying non-leaf autograd tensors requires the server')
        value = type(self)(self._data.copy(), requires_grad=self.requires_grad)
        memo[id(self)] = value
        value.grad = copy.deepcopy(self.grad, memo)
        return value

    def _binary(self, other, op, left, right):
        if not isinstance(other, (Tensor, np.ndarray)):
            if isinstance(other, (complex, np.complexfloating)):
                other = Tensor(other, dtype=np.complex128 if self.dtype in (np.dtype('float64'),np.dtype('complex128')) else np.complex64)
            elif self.dtype.kind in 'fc': other = Tensor(other, dtype=self.dtype)
        other = wrap(other)
        a, b = self._data, other._data
        floating = [value.dtype for value in (a,b) if value.dtype.kind in 'fc']
        if floating:
            dtype = np.result_type(*floating)
            for array, scalar in ((a,b),(b,a)):
                if array.ndim and not scalar.ndim and array.dtype.kind in 'fc':
                    dtype = array.dtype
                    if array.dtype.kind == 'f' and scalar.dtype.kind == 'c':
                        dtype = np.dtype('complex128' if array.dtype.itemsize == 8 else 'complex64')
            a, b = a.astype(dtype,copy=False), b.astype(dtype,copy=False)
        elif a.dtype.kind in 'biu' and b.dtype.kind in 'biu':
            dtype = a.dtype if a.ndim and not b.ndim and a.dtype.kind!='b' else b.dtype if b.ndim and not a.ndim and b.dtype.kind!='b' else np.result_type(a.dtype,b.dtype)
            a,b=a.astype(dtype,copy=False),b.astype(dtype,copy=False)
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
    def floor_divide(self, other):
        other_value = wrap(other)
        self._real_only(); other_value._real_only()
        if self.dtype.kind == other_value.dtype.kind == 'b': raise BrowserUnsupported('Boolean floor division requires the server')
        if _grad_enabled and (self.requires_grad or other_value.requires_grad):
            raise BrowserUnsupported('Floor division has no supported backward; use the server')
        if self.dtype.kind in 'biu' and other_value.dtype.kind in 'biu':
            if np.any(other_value._data == 0): raise RuntimeError('Integer division by zero')
            if self.dtype.kind == 'i' and np.any(self._data == np.iinfo(self.dtype).min) and np.any(other_value._data == -1):
                raise BrowserUnsupported('Overflowing integer floor division requires the server')
        with np.errstate(divide='ignore', invalid='ignore'):
            return self._binary(other, np.floor_divide, lambda *args: unsupported('floor_divide backward'), lambda *args: unsupported('floor_divide backward'))
    __floordiv__ = floor_divide
    def __rfloordiv__(self, other): return wrap(other).floor_divide(self)
    def remainder(self, other, *, out=None):
        if out is not None: raise BrowserUnsupported('Remainder output buffers require the server')
        self._real_only(); wrap(other)._real_only()
        if self.dtype.kind=='b' and wrap(other).dtype.kind=='b': raise BrowserUnsupported('Boolean remainder requires the server')
        if self.dtype.kind in 'biu' and wrap(other).dtype.kind in 'biu' and np.any(wrap(other)._data==0):
            raise RuntimeError('Integer division by zero')
        return self._binary(other,np.remainder,lambda g,a,b:g,lambda g,a,b:-g*np.floor(a/b))
    __mod__ = remainder
    def __rmod__(self, other):
        self._real_only(); wrap(other)._real_only()
        if self.dtype.kind=='b' and wrap(other).dtype.kind=='b': raise BrowserUnsupported('Boolean remainder requires the server')
        if self.dtype.kind in 'biu' and wrap(other).dtype.kind in 'biu' and np.any(self._data==0):
            raise RuntimeError('Integer division by zero')
        return self._binary(other,lambda a,b:np.remainder(b,a),lambda g,a,b:-g*np.floor(b/a),lambda g,a,b:g)
    def __neg__(self): return result(-self._data, (self, lambda g: -g))
    def _inplace(self, other, operation):
        if _grad_enabled and (self.requires_grad or wrap(other).requires_grad):
            raise BrowserUnsupported('In-place updates of differentiable tensors require the server')
        value = operation(self, other)
        if value.shape != self.shape or not np.can_cast(value.dtype, self.dtype, casting='same_kind'):
            raise RuntimeError('In-place operation cannot change the output shape or dtype kind')
        self[...] = value
        return self
    def __iadd__(self, other): return self._inplace(other, lambda a,b: a+b)
    def __isub__(self, other): return self._inplace(other, lambda a,b: a-b)
    def __imul__(self, other): return self._inplace(other, lambda a,b: a*b)
    def __itruediv__(self, other): return self._inplace(other, lambda a,b: a/b)
    def add_(self, other, *, alpha=1): return self._inplace(other, lambda a,b: a + b * alpha)
    def sub_(self, other, *, alpha=1): return self._inplace(other, lambda a,b: a - b * alpha)
    def mul_(self, other): return self.__imul__(other)
    def div_(self, other): return self.__itruediv__(other)
    def __pow__(self, exponent):
        if not np.isscalar(exponent): raise BrowserUnsupported('Tensor exponents require the server')
        return result(self._data ** exponent, (self, lambda g: g * exponent * self._data ** (exponent - 1)))
    def pow(self, exponent): return self ** exponent
    def square(self): return self * self
    def sqrt(self): return self ** .5
    def abs(self): return result(np.abs(self._data), (self, lambda g: g * np.sign(self._data)))
    def relu(self):
        self._real_only()
        return result(np.maximum(self._data, 0), (self, lambda g: np.where(self._data <= 0, 0, g)))
    def exp(self):
        data = np.exp(self._data)
        return result(data, (self, lambda g: g * data))
    def log(self): return result(np.log(self._data), (self, lambda g: g / self._data))
    def logit(self, eps=None):
        if self.dtype.kind != 'f': raise BrowserUnsupported('logit requires floating tensors')
        if eps is not None and (not np.isfinite(eps) or not 0 <= eps <= .5):
            raise BrowserUnsupported('logit eps outside [0, .5] requires the server')
        probability = self._data if eps is None else np.clip(self._data, eps, 1 - eps)
        with np.errstate(divide='ignore', invalid='ignore'):
            value = np.log(probability / (1 - probability))
        def vjp(g):
            with np.errstate(divide='ignore', invalid='ignore'):
                gradient = g / (probability * (1 - probability))
            if eps is None:
                gradient = np.where((self._data < 0) | (self._data > 1), np.nan, gradient)
            else:
                gradient = np.where((self._data < eps) | (self._data > 1 - eps), 0., gradient)
            return gradient
        return result(value, (self, vjp))
    def logaddexp(self, other):
        self._real_only(); wrap(other)._real_only()
        return self._binary(other, np.logaddexp,
                            lambda g,a,b: g * np.exp(-np.logaddexp(0, b-a)),
                            lambda g,a,b: g * np.exp(-np.logaddexp(0, a-b)))

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
    def reshape_as(self, other): return self.reshape(other.shape)
    def split(self, split_size_or_sections, dim=0):
        if not -self.ndim <= dim < self.ndim: raise IndexError('split dimension out of range')
        dim %= self.ndim
        size = self.shape[dim]
        if isinstance(split_size_or_sections, int):
            if split_size_or_sections <= 0 and (size or split_size_or_sections < 0): raise RuntimeError('Invalid split size')
            lengths = [min(split_size_or_sections, size - start) for start in range(0, size, split_size_or_sections or 1)] or [0]
        else:
            lengths = list(split_size_or_sections)
            if any(not isinstance(length, int) or length < 0 for length in lengths) or sum(lengths) != size: raise RuntimeError('Split sizes must sum to the dimension length')
        parts, start = [], 0
        for length in lengths:
            index = [slice(None)] * self.ndim; index[dim] = slice(start, start + length)
            parts.append(self[tuple(index)]); start += length
        return tuple(parts)
    def chunk(self, chunks, dim=0):
        if not isinstance(chunks,int) or isinstance(chunks,bool) or chunks < 1: raise RuntimeError('chunks must be a positive integer')
        if not -self.ndim <= dim < self.ndim: raise IndexError('chunk dimension out of range')
        size = self.shape[dim]
        if not size:
            index=[slice(None)]*self.ndim;index[dim]=slice(0,0)
            return tuple(self[tuple(index)] for _ in range(chunks))
        return self.split((size+chunks-1)//chunks,dim)
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
    def repeat_interleave(self, repeats, dim=None, *, output_size=None):
        source = self.flatten() if dim is None else self
        dim = 0 if dim is None else dim
        if not -source.ndim <= dim < source.ndim: raise IndexError('repeat_interleave dimension out of range')
        dim %= source.ndim
        counts = wrap(repeats)._data
        if counts.dtype.kind not in 'iu' or counts.ndim > 1 or np.any(counts < 0): raise RuntimeError('Invalid repetition counts')
        if np.any(counts > np.iinfo(np.intp).max):
            raise BrowserUnsupported('Repetition counts exceed this runtime index range')
        # wasm32 uses int32 indices even though torch.long remains int64.
        indices = np.repeat(np.arange(source.shape[dim], dtype=np.intp), counts.astype(np.intp))
        if output_size is not None and output_size != len(indices): raise RuntimeError('Incorrect repeat_interleave output_size')
        index = [slice(None)] * source.ndim; index[dim] = indices
        return source[tuple(index)]

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
    def trace(self):
        if self.ndim != 2: raise RuntimeError('trace expects a matrix')
        self._real_only()
        return result(np.trace(self._data), (self, lambda g: np.eye(*self.shape, dtype=self.dtype) * g))
    def det(self): return det(self)
    def prod(self, dim=None, keepdim=False, *, dtype=None):
        if dim is None and keepdim: raise TypeError('prod keepdim requires an explicit dimension')
        if dim is not None and not isinstance(dim,(int,np.integer)): raise BrowserUnsupported('prod requires one dimension')
        axes=reduction_axes(dim,self.ndim)
        if dtype is not None and np.dtype(dtype) != self.dtype: return self.to(dtype=dtype).prod(dim,keepdim)
        target = np.int64 if dtype is None and self.dtype.kind in 'biu' else dtype
        value=np.prod(self._data,axis=axes,keepdims=True,dtype=target)
        zeros=self._data==0
        count=zeros.sum(axis=axes,keepdims=True)
        nonzero=np.prod(np.where(zeros,1,self._data),axis=axes,keepdims=True,dtype=target)
        def vjp(g):
            if not np.isfinite(self._data).all(): raise BrowserUnsupported('Nonfinite product gradients require the server')
            if not keepdim:
                for axis in sorted(axes): g=np.expand_dims(g,axis)
            ordinary=np.divide(value,self._data,out=np.zeros_like(self._data),where=~zeros)
            derivative=np.where(count==0,ordinary,np.where((count==1)&zeros,nonzero,0))
            return g*derivative
        return result(value if keepdim else value.squeeze(axis=axes),(self,vjp))
    def amax(self, dim=None, keepdim=False, *, out=None):
        self._real_only()
        if out is not None: raise BrowserUnsupported('amax output buffers require the server')
        axes=reduction_axes(dim,self.ndim)
        value=np.max(self._data,axis=axes,keepdims=True)
        mask=self._data==value
        def vjp(g):
            if not keepdim:
                for axis in sorted(axes): g=np.expand_dims(g,axis)
            return g*mask/mask.sum(axis=axes,keepdims=True)
        return result(value if keepdim else value.squeeze(axis=axes),(self,vjp))
    def round(self, *, decimals=0, out=None):
        if decimals!=0 or out is not None: raise BrowserUnsupported('Decimal round/output buffers require the server')
        if self.dtype.kind in 'iu': return self.clone()
        if self.dtype.kind!='f': raise BrowserUnsupported('round requires real floating or integer inputs')
        return result(np.rint(self._data),(self,lambda g: np.zeros_like(self._data)))
    def norm(self, p='fro', dim=None, keepdim=False, dtype=None, *, out=None):
        if out is not None: raise BrowserUnsupported('Norm out arguments require the server')
        if p not in (2, 'fro'): raise BrowserUnsupported('Only Euclidean/Frobenius norms are supported')
        if dtype is not None: return self.to(dtype=dtype).norm(p, dim, keepdim)
        if self.dtype.kind != 'f': raise BrowserUnsupported('Norms require a floating tensor')
        axes = reduction_axes(dim, self.ndim)
        if p == 'fro' and dim is not None and len(axes) > 2:
            raise RuntimeError('Frobenius norm accepts at most two dimensions')
        lengths = np.sqrt(np.sum(self._data ** 2, axis=axes, keepdims=True))
        direction = np.divide(self._data, lengths, out=np.zeros_like(self._data), where=lengths != 0)
        def vjp(g):
            if not keepdim:
                for axis in sorted(axes): g = np.expand_dims(g, axis)
            return g * direction
        return result(lengths if keepdim else lengths.squeeze(axis=axes), (self, vjp))
    def logsumexp(self, dim, keepdim=False, *, out=None):
        if out is not None: raise BrowserUnsupported('logsumexp out arguments require the server')
        if dim is None: raise TypeError('logsumexp requires a dimension')
        if self.dtype.kind != 'f': raise BrowserUnsupported('logsumexp requires a floating tensor')
        axes = reduction_axes(dim, self.ndim)
        maximum = np.max(self._data, axis=axes, keepdims=True, initial=-np.inf)
        shift = np.where(np.isfinite(maximum), maximum, 0)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            value = np.log(np.exp(self._data - shift).sum(axis=axes, keepdims=True)) + shift
            derivative = np.exp(self._data - value)
        def vjp(g):
            if not keepdim:
                for axis in sorted(axes): g = np.expand_dims(g, axis)
            return g * derivative
        return result(value if keepdim else value.squeeze(axis=axes), (self, vjp))
    def diagonal(self, offset=0, dim1=0, dim2=1):
        if self.ndim < 2: raise RuntimeError('diagonal requires at least two dimensions')
        if not isinstance(offset, (int, np.integer)): raise TypeError('Diagonal offset must be an integer')
        axes = reduction_axes((dim1, dim2), self.ndim)
        value = np.diagonal(self._data, offset=int(offset), axis1=axes[0], axis2=axes[1])
        if self._data.flags.writeable: value.setflags(write=True)
        def vjp(g):
            gradient = np.zeros_like(self._data)
            moved = np.moveaxis(gradient, axes, (-2, -1))
            index = np.arange(value.shape[-1])
            moved[..., index + max(-int(offset), 0), index + max(int(offset), 0)] = g
            return gradient
        return result(value, (self, vjp))
    def diag(self, diagonal=0, *, out=None):
        if out is not None: raise BrowserUnsupported('diag out arguments require the server')
        if self.ndim not in (1, 2): raise RuntimeError('diag expects a vector or matrix')
        value = np.diag(self._data, k=diagonal)
        def vjp(g):
            if self.ndim == 1: return np.diag(g, k=diagonal)
            gradient = np.zeros_like(self._data)
            positions = np.arange(value.size, dtype=np.intp)
            gradient[positions + max(-diagonal, 0), positions + max(diagonal, 0)] = g
            return gradient
        return result(value, (self, vjp))
    def any(self, dim=None, keepdim=False): return Tensor(self._data.any(axis=dim, keepdims=keepdim))
    def all(self, dim=None, keepdim=False): return Tensor(self._data.all(axis=dim, keepdims=keepdim))
    def _extreme(self, dim, keepdim, maximum):
        self._real_only()
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
        self._real_only()
        value = np.argmax(self._data, axis=dim)
        if keepdim and self.ndim: value = np.reshape(value, (1,) * self.ndim) if dim is None else np.expand_dims(value, dim)
        return Tensor(value, dtype=np.int64)
    def argmin(self, dim=None, keepdim=False):
        self._real_only()
        value = np.argmin(self._data, axis=dim)
        if keepdim and self.ndim: value = np.reshape(value, (1,) * self.ndim) if dim is None else np.expand_dims(value, dim)
        return Tensor(value, dtype=np.int64)
    def argsort(self, dim=-1, descending=False, *, stable=False):
        if self.dtype.kind not in 'fiub': raise BrowserUnsupported('Sorting this dtype requires the server')
        if not isinstance(dim, (int, np.integer)) or not -max(self.ndim, 1) <= dim < max(self.ndim, 1):
            raise IndexError('argsort dimension out of range')
        if not isinstance(descending, bool) or not isinstance(stable, bool): raise TypeError('argsort flags must be boolean')
        if not self.ndim: return Tensor(0, dtype=np.int64)
        kind = 'stable' if stable else 'quicksort'
        if descending:
            # Reverse the input too: preserve equal-key order without negating
            # unsigned values or overflowing the smallest signed integer.
            indices = self.shape[dim] - 1 - np.flip(np.argsort(np.flip(self._data, axis=dim), axis=dim, kind=kind), axis=dim)
        else:
            indices = np.argsort(self._data, axis=dim, kind=kind)
        return Tensor(indices, dtype=np.int64)
    def sort(self, dim=-1, descending=False, *, stable=False, out=None):
        if out is not None: raise BrowserUnsupported('Sort output buffers require the server')
        indices = self.argsort(dim, descending, stable=stable)
        pair = namedtuple('Sort', ['values', 'indices'])
        return pair(self.gather(dim, indices) if self.ndim else self.clone(), indices)
    def topk(self, k, dim=None, largest=True, sorted=True, *, out=None):
        if out is not None: raise BrowserUnsupported('topk output buffers require the server')
        if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)): raise TypeError('topk k must be an integer')
        if not isinstance(largest, bool) or not isinstance(sorted, bool): raise TypeError('topk flags must be boolean')
        dim = -1 if dim is None else dim
        if not isinstance(dim, (int, np.integer)) or not -max(self.ndim, 1) <= dim < max(self.ndim, 1):
            raise IndexError('topk dimension out of range')
        if self.dtype.kind not in 'fiu': raise BrowserUnsupported('topk for this dtype requires the server')
        size = self.shape[dim] if self.ndim else 1
        if not 0 <= k <= size: raise RuntimeError('topk selected index k out of range')
        pair = namedtuple('TopK', ['values', 'indices'])
        if not self.ndim: return pair(self.clone(), Tensor(0, dtype=np.int64))
        # Sorted output is also valid when sorted=False. Tie identities are not
        # guaranteed by PyTorch; VJPs always follow the indices actually returned.
        ordered = self.argsort(dim, descending=largest, stable=True)
        slices = [slice(None)] * self.ndim; slices[dim] = slice(0, k)
        indices = ordered[tuple(slices)]
        return pair(self.gather(dim, indices), indices)
    def scatter(self, dim, index, src, *, reduce=None):
        if reduce is not None: raise BrowserUnsupported('Scatter reductions require the server')
        if not self.ndim: raise BrowserUnsupported('Scalar scatter requires the server')
        if not isinstance(dim, (int, np.integer)) or not -self.ndim <= dim < self.ndim: raise IndexError('scatter dimension out of range')
        dim %= self.ndim
        index = wrap(index)
        if index.dtype != np.dtype('int64'): raise BrowserUnsupported('Scatter indices must be int64')
        if index.ndim != self.ndim: raise RuntimeError('scatter index rank must match input')
        if any(index.shape[d] > self.shape[d] for d in range(self.ndim) if d != dim): raise RuntimeError('scatter index shape does not match input')
        if np.any(index._data < 0) or np.any(index._data >= self.shape[dim]): raise IndexError('scatter index out of bounds')
        if isinstance(src, Tensor):
            if src.dtype != self.dtype: raise RuntimeError('scatter source dtype must match input')
            if src.shape != index.shape: raise BrowserUnsupported('Scatter tensor sources must match index shape')
            if np.any(np.diff(np.sort(index._data, axis=dim), axis=dim) == 0):
                raise BrowserUnsupported('Scatter with duplicate tensor-source destinations requires the server')
        elif not np.isscalar(src) or np.iscomplexobj(src): raise BrowserUnsupported('This scatter source requires the server')
        coordinates = list(np.indices(index.shape)); coordinates[dim] = index._data
        coordinates = tuple(coordinates)
        data = self._data.copy(); data[coordinates] = src._data if isinstance(src, Tensor) else src
        overwritten = np.zeros(self.shape, dtype=bool); overwritten[coordinates] = True
        edges = [(self, lambda g: np.where(overwritten, 0, g))]
        if isinstance(src, Tensor): edges.append((src, lambda g: g[coordinates]))
        return result(data, *edges, saved=(index,))
    def flip(self, dims):
        if dims is None: raise TypeError('flip requires dimensions')
        axes = () if isinstance(dims, (tuple, list)) and not dims else reduction_axes(dims, self.ndim)
        return result(np.flip(self._data, axis=axes).copy(), (self, lambda g: np.flip(g, axis=axes)))
    def roll(self, shifts, dims=None):
        if dims is None and np.ndim(shifts) and len(shifts) != 1: raise RuntimeError('roll shifts and dimensions must align')
        inverse = -np.asarray(shifts)
        return result(np.roll(self._data, shifts, axis=dims), (self, lambda g: np.roll(g, inverse, axis=dims)))
    def triu(self, diagonal=0):
        if self.ndim < 2: raise RuntimeError('triu requires at least two dimensions')
        return result(np.triu(self._data, k=diagonal), (self, lambda g: np.triu(g, k=diagonal)))
    def tril(self, diagonal=0):
        if self.ndim < 2: raise RuntimeError('tril requires at least two dimensions')
        return result(np.tril(self._data, k=diagonal), (self, lambda g: np.tril(g, k=diagonal)))
    def cumsum(self, dim):
        return result(self._data.cumsum(axis=dim), (self, lambda g: np.flip(np.flip(g, axis=dim).cumsum(axis=dim), axis=dim)))
    def cumprod(self, dim, *, dtype=None):
        if not self.ndim and dim in (0, -1):
            target = dtype or (np.int64 if self.dtype.kind in 'biu' else self.dtype)
            return self.to(dtype=target)
        if not -self.ndim <= dim < self.ndim: raise IndexError('cumprod dimension out of range')
        dim %= self.ndim
        if dtype is None and self.dtype.kind in 'biu': dtype = np.int64
        value = self._data.cumprod(axis=dim, dtype=dtype)
        def vjp(g):
            x, prefix, incoming = (np.moveaxis(array, dim, -1) for array in (self._data, value, g))
            grad = np.empty_like(incoming); carry = np.zeros(incoming.shape[:-1], dtype=incoming.dtype)
            for i in reversed(range(x.shape[-1])):
                carry = incoming[..., i] + (carry * x[..., i + 1] if i + 1 < x.shape[-1] else 0)
                grad[..., i] = carry * (prefix[..., i - 1] if i else 1)
            return np.moveaxis(grad, -1, dim)
        return result(value, (self, vjp))
    def softmax(self, dim, dtype=None):
        self._real_only()
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
        self._real_only()
        if min is None and max is None: raise ValueError('clamp needs at least one bound')
        if isinstance(min, Tensor) or isinstance(max, Tensor): raise BrowserUnsupported('Tensor clamp bounds require the server')
        mask = np.ones(self.shape, dtype=bool)
        if min is not None: mask &= self._data >= min
        if max is not None: mask &= self._data <= max
        return result(np.clip(self._data, min, max), (self, lambda g: g * mask))
    def clamp_min(self, min): return self.clamp(min=min)
    def clamp_max(self, max): return self.clamp(max=max)
    def sin(self):
        source=self.float() if self.dtype.kind in 'biu' else self
        return result(np.sin(source._data), (source, lambda g: g * np.cos(source._data)))
    def cos(self):
        source=self.float() if self.dtype.kind in 'biu' else self
        return result(np.cos(source._data), (source, lambda g: -g * np.sin(source._data)))
    def acos(self): return result(np.arccos(self._data), (self, lambda g: -g / np.sqrt(1 - self._data ** 2)))
    def erfinv(self, *, out=None):
        if out is not None or self.dtype not in (np.dtype('float32'), np.dtype('float64')):
            raise BrowserUnsupported('erfinv requires real float32/float64 without output buffers')
        normal = NormalDist()
        def inverse(x):
            if math.isnan(x) or abs(x) > 1: return math.nan
            if abs(x) == 1: return math.copysign(math.inf, x)
            if x == 0: return x
            # Lower-tail evaluation avoids rounding (1+x)/2 to one near x=1.
            value = math.copysign(-normal.inv_cdf((1-abs(x))/2) / math.sqrt(2), x)
            # Near zero the quantile's probability can round to 0.5. A central
            # erf correction retains tiny inputs without cancellation at tails.
            if abs(x) < .5: value -= (math.erf(value)-x) * (math.sqrt(math.pi)/2) * math.exp(value*value)
            return value
        value = np.fromiter((inverse(float(x)) for x in self._data.flat), dtype=self.dtype, count=self.numel()).reshape(self.shape)
        return result(value, (self, lambda g: g * (math.sqrt(math.pi)/2) * np.exp(value**2)))
    def tanh(self):
        value = np.tanh(self._data)
        return result(value, (self, lambda g: g * (1 - value ** 2)))
    def rsqrt(self): return self ** -.5
    def sigmoid(self):
        self._real_only()
        exp = np.exp(-np.abs(self._data))
        value = np.where(self._data >= 0, 1 / (1 + exp), exp / (1 + exp))
        return result(value, (self, lambda g: g * value * (1 - value)))
    def logsigmoid(self):
        self._real_only()
        zero = np.zeros((), dtype=self.dtype)
        return result(-np.logaddexp(zero, -self._data), (self, lambda g: g * np.exp(-np.logaddexp(zero, self._data))))
    def log_softmax(self, dim, dtype=None):
        self._real_only()
        x = self.to(dtype=dtype) if dtype is not None else self
        shifted = x._data - x._data.max(axis=dim, keepdims=True)
        value = shifted - np.log(np.exp(shifted).sum(axis=dim, keepdims=True))
        return result(value, (x, lambda g: g - np.exp(value) * g.sum(axis=dim, keepdims=True)))
    def gather(self, dim, index):
        dim %= self.ndim
        saved_index = wrap(index)
        index = saved_index._data
        if index.ndim != self.ndim or index.dtype.kind not in 'iu': raise ValueError('gather needs integer indices of the same rank')
        if np.any(index < 0) or np.any(index >= self.shape[dim]): raise IndexError('gather index out of bounds')
        if any(index.shape[d] > self.shape[d] for d in range(self.ndim) if d != dim): raise ValueError('gather dimensions do not match')
        slices = tuple(slice(None) if d == dim else slice(0, index.shape[d]) for d in range(self.ndim))
        def vjp(g):
            coordinates = list(np.indices(index.shape)); coordinates[dim] = index
            grad = np.zeros_like(self._data); np.add.at(grad, tuple(coordinates), g)
            return grad
        return result(np.take_along_axis(self._data[slices], index, axis=dim), (self, vjp), saved=(saved_index,))
    def var(self, dim=None, unbiased=True, keepdim=False, *, correction=None):
        self._real_only()
        dims = tuple(range(self.ndim)) if dim is None else tuple(d % self.ndim for d in (dim if isinstance(dim, (list, tuple)) else (dim,)))
        count = int(np.prod([self.shape[d] for d in dims]))
        correction = int(unbiased) if correction is None else correction
        centered = self - self.mean(dim, keepdim=True)
        return centered.square().sum(dim, keepdim) / (count - correction)
    def std(self, dim=None, unbiased=True, keepdim=False, *, correction=None):
        self._real_only()
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

    def _compare(self, other, op):
        if isinstance(other,(complex,np.complexfloating)): raise BrowserUnsupported('Complex comparisons require the server')
        # Python scalars use the floating tensor's precision, just like arithmetic.
        # Wrapping first would irreversibly round a float64 boundary to float32.
        if not isinstance(other, (Tensor, np.ndarray)) and self.dtype.kind == 'f':
            other = Tensor(other, dtype=self.dtype)
        other = wrap(other)
        self._real_only(); other._real_only()
        a, b = self._data, other._data
        if a.dtype.kind == 'f' and b.dtype.kind == 'f':
            dtype = a.dtype if a.ndim and not b.ndim else b.dtype if b.ndim and not a.ndim else np.result_type(a.dtype, b.dtype)
            a, b = a.astype(dtype, copy=False), b.astype(dtype, copy=False)
        return Tensor(op(a, b))
    def __eq__(self, other): return self._compare(other, np.equal)
    def __ne__(self, other): return self._compare(other, np.not_equal)
    def __lt__(self, other): return self._compare(other, np.less)
    def __le__(self, other): return self._compare(other, np.less_equal)
    def __gt__(self, other): return self._compare(other, np.greater)
    def __ge__(self, other): return self._compare(other, np.greater_equal)
    eq, ne, lt, le, gt, ge = __eq__, __ne__, __lt__, __le__, __gt__, __ge__
    def __invert__(self): return Tensor(~self._data)
    def __and__(self, other): return Tensor(self._data & wrap(other)._data)
    def __or__(self, other): return Tensor(self._data | wrap(other)._data)


def count_nonzero(input, dim=None):
    axes = None if dim is None else reduction_axes(dim, input.ndim)
    return Tensor(np.count_nonzero(input._data, axis=axes), dtype=np.int64)


def where(condition, x, y):
    condition, a, b = wrap(condition), wrap(x), wrap(y)
    mask = condition._data
    return result(np.where(mask, a._data, b._data),
                  (a, lambda g: unbroadcast(np.where(mask, g, 0), a.shape)),
                  (b, lambda g: unbroadcast(np.where(mask, 0, g), b.shape)), saved=(condition,))


def tensor(data, dtype=None, *, device=None, requires_grad=False):
    check_device(device)
    if isinstance(data, Tensor): data = data._data.copy()
    array = np.asarray(data)
    if dtype is None and not isinstance(data, (np.ndarray, np.generic)):
        dtype = np.float32 if array.dtype.kind == 'f' else np.complex64 if array.dtype.kind == 'c' else np.int64 if array.dtype.kind in 'iu' else None
    return Tensor(array, dtype=dtype, requires_grad=requires_grad)


def check_device(device):
    if device not in (None, 'cpu', 'cpu:0'): raise BrowserUnsupported('Only CPU/WASM tensors are supported')


@contextlib.contextmanager
def fork_rng(devices=None, enabled=True, device_type='cuda'):
    if devices and enabled: raise BrowserUnsupported('Non-CPU RNG states require the server')
    if device_type not in ('cpu', 'cuda'): raise BrowserUnsupported('This RNG device type requires the server')
    if not enabled:
        yield
        return
    state, normal = np.random.get_state(), _default_generator.normal
    try: yield
    finally:
        np.random.set_state(state)
        _default_generator.normal = normal


class Generator:
    def __init__(self, device=None):
        check_device(device)
        self.rng, self.normal = np.random.RandomState(42), None
    def manual_seed(self, seed):
        if self.rng is np.random: np.random.seed(seed)
        else: self.rng = np.random.RandomState(seed)
        self.normal = None
        return self


_default_generator = Generator()
_default_generator.rng = np.random


def normal_samples(generator, shape, dtype):
    """CPU Box-Muller filling and cached scalar draws, matching PyTorch's stream."""
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise BrowserUnsupported('Normal sampling requires float32/float64')
    size = math.prod(shape)
    rng = generator.rng
    if size < 16:
        values = np.empty(size, dtype=dtype)
        for i in range(size):
            if generator.normal is None:
                u1, u2 = uniform_samples(rng, (2,), np.float64)
                radius, theta = math.sqrt(-2 * math.log1p(-u2)), 2 * math.pi * u1
                value, generator.normal = radius * math.cos(theta), radius * math.sin(theta)
            else:
                value, generator.normal = generator.normal, None
            values[i] = value
        return values.reshape(shape)
    values = uniform_samples(rng, (size,), dtype)
    def transform(blocks):
        radius = np.sqrt(-2 * np.log(1 - blocks[:, :8]))
        theta = (2 * math.pi * blocks[:, 8:].astype(np.float64)).astype(dtype)
        blocks[:, :8], blocks[:, 8:] = radius * np.cos(theta), radius * np.sin(theta)
    transform(values[:size // 16 * 16].reshape(-1, 16))
    if size % 16:
        values[-16:] = uniform_samples(rng, (16,), dtype)
        transform(values[-16:].reshape(1, 16))
    return values.reshape(shape)


def multinomial(probabilities, num_samples, replacement=False, *, generator=None):
    weights = wrap(probabilities)._data
    if not replacement and num_samples != 1:
        raise BrowserUnsupported('Multiple draws without replacement require the server')
    if weights.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise BrowserUnsupported('Multinomial supports float32/float64 probabilities')
    if weights.ndim not in (1, 2) or not weights.shape[-1] or type(num_samples) is not int or num_samples < 1:
        raise RuntimeError('Invalid multinomial shape or sample count')
    if not np.isfinite(weights).all() or np.any(weights < 0): raise RuntimeError('Invalid multinomial probabilities')
    rng = generator.rng if generator else np.random
    if num_samples == 1:
        if np.any(weights.sum(axis=-1) <= 0): raise RuntimeError('Multinomial probabilities must sum to a positive value')
        # Independent exponential races implement the single-draw categorical law.
        # This is distribution-faithful; its seeded stream is not native PyTorch's.
        uniforms = uniform_samples(rng, weights.shape, weights.dtype)
        with np.errstate(divide='ignore', invalid='ignore'):
            races = -np.log1p(-uniforms) / weights
        races = np.where(weights > 0, races, np.inf)
        return Tensor(np.argmin(races, axis=-1)[..., None], dtype=np.int64)
    draws = []
    for row in weights.reshape(-1, weights.shape[-1]):
        cumulative = np.cumsum(row, dtype=weights.dtype)
        if not cumulative[-1] > 0: raise RuntimeError('Multinomial probabilities must sum to a positive value')
        cumulative /= cumulative[-1]; cumulative[-1] = 1
        # Native CPU replacement sampling consumes float64 uniform draws even
        # when the probabilities themselves are float32.
        draws.append(np.searchsorted(cumulative, uniform_samples(rng, (num_samples,), np.float64), side='left'))
    return Tensor(np.asarray(draws, dtype=np.int64).reshape(weights.shape[:-1] + (num_samples,)))


def randperm(n, *, generator=None, dtype=np.int64, device=None):
    check_device(device)
    if type(n) is not int or n < 0: raise RuntimeError('randperm needs a non-negative integer')
    if n > 1_000_000 or np.dtype(dtype) != np.dtype('int64'):
        raise BrowserUnsupported('Large/non-int64 permutations require the server')
    rng = generator.rng if generator else np.random
    values = np.arange(n, dtype=np.int64)
    bits = rng.randint(0, 2**32, size=max(0, n-1), dtype=np.uint32)
    for i, bit in enumerate(bits):
        other = i + int(bit) % (n-i)
        values[i], values[other] = values[other], values[i]
    return Tensor(values)


def elementwise_extreme(a, b, maximum=False):
    a = wrap(a)
    a._real_only(); wrap(b)._real_only()
    def share(left, right):
        blocked = left < right if maximum else left > right
        return np.where(left == right, .5, np.where(blocked, 0., 1.))
    return a._binary(b, np.maximum if maximum else np.minimum,
                     lambda g,x,y: g * share(x,y), lambda g,x,y: g * share(y,x))


def solve(a, b):
    a, b = wrap(a), wrap(b)
    if a.ndim != 2 or b.ndim not in (1, 2): raise BrowserUnsupported('Batched linear solves require the server')
    if a.dtype != b.dtype or a.dtype.kind != 'f': raise RuntimeError('solve expects matching floating dtypes')
    value = np.linalg.solve(a._data, b._data)
    def db(g): return np.linalg.solve(a._data.T, g)
    def da(g):
        incoming = db(g)
        return -(incoming[:, None] @ value[None, :] if b.ndim == 1 else incoming @ value.T)
    return result(value, (a, da), (b, db))


def inv(input, *, out=None):
    if out is not None: raise BrowserUnsupported('Inverse output buffers require the server')
    real_matrix(input)
    if input.shape[-1] != input.shape[-2]: raise RuntimeError('Inverse expects square matrices')
    try: value = np.linalg.inv(input._data)
    except np.linalg.LinAlgError as error: raise RuntimeError('The matrix is singular') from error
    transpose = np.swapaxes(value, -1, -2)
    return result(value, (input, lambda g: -transpose @ g @ transpose))


def trapezoid(y, x=None, *, dx=None, dim=-1):
    y = wrap(y)
    if not y.ndim or not isinstance(dim, (int,np.integer)) or not -y.ndim <= dim < y.ndim:
        raise IndexError('trapezoid dimension out of range')
    if y.dtype.kind not in 'fc': raise BrowserUnsupported('Integer trapezoid inputs require the server')
    dim %= y.ndim
    if x is not None and dx is not None: raise TypeError('Specify x or dx, not both')
    if y.shape[dim] == 0:
        return Tensor(np.zeros(y.shape[:dim] + y.shape[dim+1:],dtype=y.dtype))
    left, right = [slice(None)]*y.ndim, [slice(None)]*y.ndim
    left[dim], right[dim] = slice(None,-1), slice(1,None)
    average = (y[tuple(left)] + y[tuple(right)]) / 2
    if x is None:
        if dx is not None and (not np.isscalar(dx) or isinstance(dx,complex)):
            raise BrowserUnsupported('trapezoid dx must be a real scalar')
        return (average * (1 if dx is None else dx)).sum(dim)
    x = wrap(x)
    if not x.ndim or x.ndim > y.ndim or x.dtype.kind not in 'fc':
        raise BrowserUnsupported('trapezoid coordinates require floating tensors of supported rank')
    if x.ndim == 1:
        if x.numel() != y.shape[dim]: raise RuntimeError('trapezoid coordinate length does not match samples')
        shape = [1]*y.ndim; shape[dim] = x.numel(); x = x.reshape(shape)
    else: x = x.reshape((1,)*(y.ndim-x.ndim) + x.shape)
    return (average * (x[tuple(right)] - x[tuple(left)])).sum(dim)


def real_matrix(input):
    if input.ndim < 2: raise RuntimeError('Linear algebra expects matrices')
    if input.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise BrowserUnsupported('This linear algebra backend requires real float32/float64 matrices')
    if not np.isfinite(input._data).all(): raise RuntimeError('Linear algebra requires finite inputs')


def svd(input, full_matrices=True, *, driver=None, out=None):
    """Real batched SVD with analytical vector/rectangular-complement VJPs."""
    real_matrix(input)
    if driver is not None or out is not None: raise BrowserUnsupported('SVD drivers/output buffers require the server')
    if not isinstance(full_matrices, bool): raise TypeError('full_matrices must be boolean')
    all_u, singular, all_vh = np.linalg.svd(input._data, full_matrices=full_matrices)
    rank = singular.shape[-1]
    u, vh = all_u[..., :, :rank], all_vh[..., :rank, :]
    ut, v = np.swapaxes(u,-1,-2), np.swapaxes(vh,-1,-2)
    saved = tuple(Tensor(value) for value in (all_u, singular, all_vh))
    def ds(g): return (u * g[..., None, :]) @ vh
    def vector_factors():
        # Do not fabricate derivatives at rank boundaries or repeated spectra.
        with np.errstate(over='ignore',invalid='ignore'):
            squared = singular ** 2
            gap = squared[..., None, :] - squared[..., :, None]
        diagonal = np.eye(rank, dtype=bool)
        cutoff = np.finfo(input.dtype).eps * max(input.shape[-2:])
        if not np.isfinite(squared).all() or np.any(singular <= cutoff * singular[..., :1]) or np.any((np.abs(gap) <= cutoff * squared[..., :1, None]) & ~diagonal):
            raise BrowserUnsupported('Degenerate or numerically unresolved SVD vector gradients require the server')
        return np.divide(1, gap, out=np.zeros_like(gap), where=~diagonal)
    def du(g):
        if not rank: return np.zeros_like(input._data)
        g = g[..., :, :rank]
        inner = ut @ g
        core = vector_factors() * (inner - np.swapaxes(inner,-1,-2)) * singular[..., None, :]
        gradient = u @ core @ vh
        if input.shape[-2] > rank: gradient += ((g - u @ inner) / singular[..., None, :]) @ vh
        return gradient
    def dvh(g):
        if not rank: return np.zeros_like(input._data)
        g = g[..., :rank, :]
        inner = vh @ np.swapaxes(g,-1,-2)
        core = vector_factors() * (inner - np.swapaxes(inner,-1,-2)) * singular[..., :, None]
        gradient = u @ core @ vh
        if input.shape[-1] > rank: gradient += u @ ((g - (g @ v) @ vh) / singular[..., :, None])
        return gradient
    pair = namedtuple('SVD', ['U','S','Vh'])
    return pair(result(all_u,(input,du),saved=saved), result(singular,(input,ds),saved=saved), result(all_vh,(input,dvh),saved=saved))


def eigh(input, UPLO='L', *, out=None):
    """Real symmetric eigendecomposition with separate value/vector VJPs."""
    real_matrix(input)
    if out is not None: raise BrowserUnsupported('eigh output buffers require the server')
    if input.shape[-2] != input.shape[-1] or UPLO not in ('L','U'): raise RuntimeError('Invalid eigh shape/triangle')
    values, vectors = np.linalg.eigh(input._data, UPLO=UPLO)
    count = input.shape[-1]
    transposed = np.swapaxes(vectors, -1, -2)
    saved = (Tensor(values), Tensor(vectors))
    def spectral_gaps():
        with np.errstate(over='ignore', invalid='ignore'):
            gaps = values[..., None, :] - values[..., :, None]
        scale = np.max(np.abs(values), axis=-1, keepdims=True, initial=0)
        tolerance = np.finfo(input.dtype).eps * max(count, 1) * scale[..., None]
        unresolved = (np.abs(gaps) <= tolerance) & ~np.eye(count, dtype=bool)
        return gaps, unresolved
    def dl(g):
        _, unresolved = spectral_gaps()
        # At repeated spectra a differentiable spectral objective has equal
        # derivatives within each eigenspace. Otherwise the chosen basis would
        # invent a platform-dependent gradient.
        differences = np.abs(g[..., None, :] - g[..., :, None])
        tolerance = 8 * np.finfo(input.dtype).eps * np.maximum(1, np.max(np.abs(g), axis=-1, keepdims=True, initial=0))[..., None]
        if np.any(unresolved & (differences > tolerance)):
            raise BrowserUnsupported('Basis-dependent repeated-eigenvalue gradients require the server')
        return (vectors * g[..., None, :]) @ transposed
    def dq(g):
        gaps, unresolved = spectral_gaps()
        if np.any(unresolved) or not np.isfinite(gaps).all():
            raise BrowserUnsupported('Degenerate or unresolved eigenvector gradients require the server')
        inner = transposed @ g
        core = .5 * np.divide(inner - np.swapaxes(inner, -1, -2), gaps,
                              out=np.zeros_like(inner), where=~np.eye(count, dtype=bool))
        # PyTorch assumes symmetric input: the VJP is symmetric, even when
        # UPLO selected just one triangle for the forward computation.
        return vectors @ core @ transposed
    return namedtuple('Eigh',['eigenvalues','eigenvectors'])(
        result(values, (input, dl), saved=saved), result(vectors, (input, dq), saved=saved))


def eigvalsh(input, UPLO='L', *, out=None):
    if out is not None: raise BrowserUnsupported('eigvalsh output buffers require the server')
    return eigh(input, UPLO).eigenvalues


def det(input, *, out=None):
    real_matrix(input)
    if out is not None: raise BrowserUnsupported('Determinant output buffers require the server')
    if input.shape[-2] != input.shape[-1]: raise RuntimeError('det expects square matrices')
    value = np.linalg.det(input._data)
    def vjp(g):
        # Products of all other singular values give the cofactor matrix
        # without dividing by a singular determinant or singular value.
        u, s, vh = np.linalg.svd(input._data, full_matrices=False)
        ones = np.ones(s.shape[:-1] + (1,), dtype=s.dtype)
        prefix = np.concatenate((ones, np.cumprod(s, axis=-1)), axis=-1)
        suffix = np.concatenate((np.flip(np.cumprod(np.flip(s, axis=-1), axis=-1), axis=-1), ones), axis=-1)
        cofactors = prefix[..., :-1] * suffix[..., 1:]
        orientation = np.linalg.det(u) * np.linalg.det(vh)
        return (g * orientation)[..., None, None] * ((u * cofactors[..., None, :]) @ vh)
    return result(value, (input, vjp))


def qr(input, mode='reduced', *, out=None):
    real_matrix(input)
    if out is not None or mode not in ('reduced','complete') or _grad_enabled and input.requires_grad:
        raise BrowserUnsupported('Differentiable/other-mode QR or output buffers require the server')
    q, r = np.linalg.qr(input._data, mode=mode)
    return namedtuple('QR',['Q','R'])(Tensor(q),Tensor(r))


def linalg_norm(input, ord=None, dim=None, keepdim=False, *, out=None, dtype=None):
    if out is not None: raise BrowserUnsupported('Norm output buffers require the server')
    if dtype is not None and (np.dtype(dtype).kind != 'f' or np.dtype(dtype).itemsize < input.dtype.itemsize):
        raise BrowserUnsupported('Narrowing/non-floating norm dtypes require the server')
    axes = reduction_axes(dim, input.ndim)
    if dim is not None and len(axes) not in (1, 2): raise RuntimeError('linalg.norm requires one or two dimensions')
    if ord is not None:
        if dim is None and input.ndim not in (1, 2): raise RuntimeError('An explicit norm order requires a vector or matrix')
        if ord == 'fro' and len(axes) != 2: raise RuntimeError('Frobenius norm requires two dimensions')
        if ord != 'fro' and (ord != 2 or len(axes) != 1):
            raise BrowserUnsupported('Other vector norms and spectral matrix norms require the server')
    return input.norm(2, dim, keepdim, dtype)


class Parameter(Tensor):
    def __init__(self, data=None, requires_grad=True):
        super().__init__(np.empty(0, dtype=np.float32) if data is None else wrap(data)._data, requires_grad=requires_grad)


class Module:
    """Direct Parameters and nested Modules; other module interfaces fall back."""
    def __init__(self): self.training, self._buffers = True, {}
    def __getattr__(self, name):
        if name.startswith('_'): raise AttributeError(name)
        return unsupported(f'nn.Module.{name}')
    def __call__(self, *args, **kw): return self.forward(*args, **kw)
    def parameters(self, recurse=True):
        seen, modules, pending = set(), set(), [self]
        while pending:
            module = pending.pop()
            if id(module) in modules: continue
            modules.add(id(module))
            children = []
            for name, value in module.__dict__.items():
                if name in module._buffers: continue
                if isinstance(value, Parameter) and id(value) not in seen:
                    seen.add(id(value)); yield value
                elif recurse and isinstance(value, Module): children.append(value)
            pending.extend(reversed(children))

    def requires_grad_(self, requires_grad=True):
        for parameter in self.parameters(): parameter.requires_grad_(requires_grad)
        return self

    def register_buffer(self, name, tensor, persistent=True):
        if not isinstance(name, str) or not name or '.' in name: raise ValueError('Buffer name must be nonempty without dots')
        if tensor is not None and not isinstance(tensor, Tensor): raise TypeError('Buffer must be a Tensor or None')
        if name in self.__dict__ and name not in self._buffers: raise KeyError('Attribute already exists')
        self._buffers[name] = tensor
        setattr(self, name, tensor)


def linear(input, weight, bias=None):
    if input.ndim < 1 or weight.ndim != 2 or input.shape[-1] != weight.shape[1] or input.dtype != weight.dtype:
        raise RuntimeError('linear expects matching feature dimensions and dtypes')
    if input.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise BrowserUnsupported('linear requires float32/float64')
    if bias is not None and (bias.shape != (weight.shape[0],) or bias.dtype != input.dtype):
        raise BrowserUnsupported('Other linear bias shapes/dtypes require the server')
    rows = math.prod(input.shape[:-1])
    x, w = input._data, weight._data
    value = x @ w.T
    if bias is not None: value = value + bias._data
    edges = [(input, lambda g: g @ w),
             (weight, lambda g: g.reshape(rows, w.shape[0]).T @ x.reshape(rows, w.shape[1]))]
    if bias is not None: edges.append((bias, lambda g: g.reshape(rows, w.shape[0]).sum(axis=0)))
    return result(value, *edges)


def silu(input, inplace=False):
    input._real_only()
    if inplace: raise BrowserUnsupported('In-place SiLU requires the server')
    x = input._data
    exp = np.exp(-np.abs(x))
    probability = np.where(x >= 0, 1 / (1 + exp), exp / (1 + exp))
    return result(x * probability, (input, lambda g: g * probability * (1 + x * (1 - probability))))


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        check_device(device)
        dtype = np.dtype(np.float32 if dtype is None else dtype)
        if dtype not in (np.dtype('float32'), np.dtype('float64')):
            raise BrowserUnsupported('Linear requires float32/float64')
        if type(in_features) is not int or type(out_features) is not int or min(in_features, out_features) < 0:
            raise ValueError('Linear dimensions must be nonnegative integers')
        self.in_features, self.out_features = in_features, out_features
        bound = in_features ** -.5 if in_features else 0
        def parameter(shape):
            return Parameter((uniform_samples(np.random, shape, dtype) * (2 * bound) - bound).astype(dtype))
        self.weight = parameter((out_features, in_features))
        self.bias = parameter((out_features,)) if bias else None

    def forward(self, input):
        return linear(input, self.weight, self.bias)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        if any(not isinstance(module, Module) for module in modules):
            raise BrowserUnsupported('Sequential supports positional Module instances')
        self._sequence_names = tuple(str(i) for i in range(len(modules)))
        for name, module in zip(self._sequence_names, modules): setattr(self, name, module)

    def __iter__(self): return (getattr(self, name) for name in self._sequence_names)
    def __len__(self): return len(self._sequence_names)
    def __getitem__(self, index):
        values = tuple(self)
        return Sequential(*values[index]) if isinstance(index, slice) else values[index]
    def forward(self, input):
        for module in self: input = module(input)
        return input


class SiLU(Module):
    def __init__(self, inplace=False):
        super().__init__()
        if inplace: raise BrowserUnsupported('In-place SiLU requires the server')
    def forward(self, input): return silu(input)


class ReLU(Module):
    def __init__(self, inplace=False):
        super().__init__()
        if inplace: raise BrowserUnsupported('In-place ReLU requires the server')
    def forward(self, input): return input.relu()


def bincount(input, weights=None, minlength=0):
    if weights is not None: raise BrowserUnsupported('Weighted bincount requires the server')
    if input.ndim != 1 or input.dtype.kind not in 'iu' or np.any(input._data < 0) or type(minlength) is not int or minlength < 0:
        raise RuntimeError('bincount expects a nonnegative integer vector and minlength')
    return Tensor(np.bincount(input._data.astype(np.int64), minlength=minlength), dtype=np.int64)


def assert_close(actual, expected, *, rtol=None, atol=None, equal_nan=False, check_device=True, check_dtype=True, **options):
    if options: raise BrowserUnsupported('Advanced torch.testing options require the server')
    actual, expected = wrap(actual), wrap(expected)
    if actual.shape != expected.shape: raise AssertionError(f'Shapes differ: {actual.shape} != {expected.shape}')
    if check_dtype and actual.dtype != expected.dtype: raise AssertionError(f'Dtypes differ: {actual.dtype} != {expected.dtype}')
    if (rtol is None) != (atol is None): raise ValueError('Specify both rtol and atol')
    if rtol is None:
        defaults = {np.dtype('float32'): (1.3e-6, 1e-5), np.dtype('float64'): (1e-7, 1e-7)}
        pairs = [defaults.get(value.dtype, (0., 0.)) for value in (actual, expected)]
        rtol, atol = (max(pair[i] for pair in pairs) for i in (0, 1))
    if rtol == 0 and atol == 0 and actual.dtype.kind in 'biu' and expected.dtype.kind in 'biu':
        np.testing.assert_array_equal(actual._data, expected._data)
    else:
        np.testing.assert_allclose(actual._data, expected._data, rtol=rtol, atol=atol, equal_nan=equal_nan)


def create(kind, *shape, dtype=np.float32, device=None, requires_grad=False, generator=None):
    check_device(device)
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)): shape = tuple(shape[0])
    rng = generator.rng if generator else np.random
    value = normal_samples(generator or _default_generator, shape, dtype) if kind == 'randn' else uniform_samples(rng, shape, dtype) if kind == 'rand' else getattr(np, kind)(shape, dtype=dtype)
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


def gradcheck(func, inputs, *, eps=1e-6, atol=1e-5, rtol=1e-3, raise_exception=True, **options):
    """Bounded full first-order Jacobian check; never substitutes training VJPs."""
    if options: raise BrowserUnsupported('Advanced gradcheck options require the server')
    inputs = tuple(inputs) if isinstance(inputs, (list, tuple)) else (inputs,)
    if any(not isinstance(x, Tensor) or x.dtype != np.dtype('float64') for x in inputs):
        raise BrowserUnsupported('gradcheck requires real float64 tensor inputs')
    active = [i for i,x in enumerate(inputs) if x.requires_grad]
    if not active: raise ValueError('gradcheck requires a differentiable input')
    if sum(inputs[i].numel() for i in active) > 256:
        raise BrowserUnsupported('Large gradcheck inputs require the server')
    for i in active:
        x = inputs[i]
        offsets = {sum(index * stride for index,stride in zip(position, x._data.strides)) for position in np.ndindex(x.shape)}
        if len(offsets) != x.numel(): raise BrowserUnsupported('Overlapping gradcheck storage requires the server')
    if not np.isfinite([eps, atol, rtol]).all() or eps <= 0 or min(atol, rtol) < 0:
        raise ValueError('Invalid gradcheck tolerances')
    def evaluate(values):
        output = func(*values)
        output = tuple(output) if isinstance(output, (list, tuple)) else (output,)
        if not output or any(not isinstance(x, Tensor) or x.dtype != np.dtype('float64') for x in output):
            raise BrowserUnsupported('gradcheck requires real float64 tensor outputs')
        if sum(x.numel() for x in output) > 256: raise BrowserUnsupported('Large gradcheck outputs require the server')
        return output
    # Cloned independent leaves prevent perturbations or .grad accumulation in
    # the caller. Reject shared input storage rather than mis-check aliases.
    if any(np.shares_memory(a._data, b._data) for i,a in enumerate(inputs) for b in inputs[i+1:]):
        raise BrowserUnsupported('Overlapping gradcheck inputs require the server')
    leaves = [Tensor(x._data.copy(), requires_grad=x.requires_grad) for x in inputs]
    outputs = evaluate(leaves)
    size = sum(x.numel() for x in outputs)
    analytical = [np.zeros((size, leaves[i].numel())) for i in active]
    row = 0
    for output in outputs:
        for index in range(output.numel()):
            if output.requires_grad:
                seed = np.zeros(output.shape); seed.flat[index] = 1
                derivatives = autograd_grad(output, [leaves[i] for i in active], Tensor(seed), allow_unused=True)
                for matrix, derivative in zip(analytical, derivatives):
                    if derivative is not None: matrix[row] = derivative._data.reshape(-1)
            row += 1
    def numerical_output(values):
        with no_grad(): evaluated = evaluate(values)
        if len(evaluated) != len(outputs) or any(a.shape != b.shape for a,b in zip(evaluated, outputs)):
            raise RuntimeError('gradcheck output shape changed during perturbation')
        return np.concatenate([x._data.reshape(-1) for x in evaluated])
    baseline = np.concatenate([x._data.reshape(-1) for x in outputs])
    success = np.allclose(numerical_output(leaves), baseline, atol=atol, rtol=rtol)
    for matrix, index in zip(analytical, active):
        numerical = np.empty_like(matrix)
        for column in range(leaves[index].numel()):
            plus = [Tensor(x._data.copy(), requires_grad=x.requires_grad) for x in leaves]
            minus = [Tensor(x._data.copy(), requires_grad=x.requires_grad) for x in leaves]
            plus[index]._data.flat[column] += eps
            minus[index]._data.flat[column] -= eps
            numerical[:,column] = (numerical_output(plus) - numerical_output(minus)) / (2 * eps)
        success = bool(success and np.allclose(matrix, numerical, atol=atol, rtol=rtol))
    if not success and raise_exception: raise RuntimeError('gradcheck Jacobian mismatch between analytical and numerical derivatives')
    return bool(success)


class no_grad(contextlib.ContextDecorator):
    def __enter__(self):
        global _grad_enabled
        self.before = _grad_enabled; _grad_enabled = False
        return self
    def __exit__(self, *args):
        global _grad_enabled
        _grad_enabled = self.before


class SGD:
    def __init__(self, params, lr=.001, momentum=0, dampening=0, weight_decay=0, nesterov=False, *, maximize=False, **options):
        if options: raise BrowserUnsupported('Advanced SGD options require the server')
        if lr < 0 or momentum < 0 or weight_decay < 0: raise ValueError('Invalid SGD hyperparameter')
        if nesterov and (momentum <= 0 or dampening != 0): raise ValueError('Nesterov requires momentum and zero dampening')
        self.params, self.lr, self.momentum = list(params), lr, momentum
        if any(not isinstance(p, Tensor) for p in self.params): raise BrowserUnsupported('SGD parameter groups require the server')
        self.dampening, self.weight_decay, self.nesterov, self.maximize = dampening, weight_decay, nesterov, maximize
        self.buffers = [None] * len(self.params)

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            if p.grad is not None: p.grad = None if set_to_none else Tensor(np.zeros_like(p._data))

    def step(self, closure=None):
        if closure is not None: raise BrowserUnsupported('SGD closures require the server')
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
            p._storage.version += 1


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
                p._storage.version += 1
                for value in state.values(): value._storage.version += 1


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


def rms_norm_oracle(x, normalized_shape, weight=None, eps=None):
    """Independent raw NumPy RMS normalization with analytical input/scale VJPs."""
    shape = tuple(normalized_shape)
    if not shape or x.shape[-len(shape):] != shape: raise RuntimeError('Invalid RMSNorm shape')
    if x.dtype.kind != 'f': raise BrowserUnsupported('RMSNorm requires floating tensors')
    if weight is not None and (weight.shape != shape or weight.dtype != x.dtype):
        raise BrowserUnsupported('Mixed-dtype RMSNorm weights require the server')
    axes = tuple(range(x.ndim-len(shape), x.ndim))
    inverse = 1 / np.sqrt(np.mean(x._data ** 2, axis=axes, keepdims=True) + (np.finfo(x.dtype).eps if eps is None else eps))
    normalized = x._data * inverse
    scale = 1 if weight is None else weight._data
    def dx(g):
        scaled = g * scale
        return inverse * (scaled - normalized * (scaled * normalized).mean(axis=axes, keepdims=True))
    edges = [(x, dx)]
    if weight is not None: edges.append((weight, lambda g: unbroadcast(g * normalized, weight.shape)))
    return result(normalized * scale, *edges)


def cov(x, *, correction=1, fweights=None, aweights=None):
    if fweights is not None or aweights is not None:
        raise BrowserUnsupported('Weighted covariance requires the server')
    if x.ndim > 2 or x.dtype.kind != 'f': raise BrowserUnsupported('Covariance requires a floating scalar/vector/matrix')
    if not isinstance(correction, int) or correction < 0: raise ValueError('Invalid covariance correction')
    rows = x._data.reshape(1, -1) if x.ndim < 2 else x._data
    count = rows.shape[-1] - correction
    if count <= 0: raise BrowserUnsupported('Insufficient covariance observations require the server')
    centered = rows - rows.mean(axis=1, keepdims=True)
    value = centered @ centered.T / count
    def vjp(g):
        matrix = np.asarray(g).reshape(value.shape)
        grad = (matrix + matrix.T) @ centered / count
        return (grad - grad.mean(axis=1, keepdims=True)).reshape(x.shape)
    return result(value.squeeze(), (x, vjp))


def randint(low=0, high=None, size=None, *, generator=None, dtype=np.int64, device=None):
    check_device(device)
    if size is None and isinstance(high, (tuple, list)):
        size, high, low = high, low, 0
    elif high is None:
        high, low = low, 0
    if not isinstance(low, (int, np.integer)) or not isinstance(high, (int, np.integer)):
        raise TypeError('randint bounds must be integers')
    low, high = int(low), int(high)
    if not isinstance(size, (tuple, list)) or any(not isinstance(s, (int, np.integer)) for s in size):
        raise TypeError('randint requires an integer shape')
    size = tuple(int(s) for s in size)
    if any(s < 0 for s in size): raise RuntimeError('randint dimensions must be nonnegative')
    if math.prod(size) > 16_000_000: raise BrowserUnsupported('randint allocation exceeds the browser limit')
    if dtype is None: dtype = np.int64
    dtype = np.dtype(dtype)
    if dtype.kind not in 'biuf': raise BrowserUnsupported('This randint dtype requires the server')
    if dtype.kind in 'biu':
        lower, upper = (0, 1) if dtype.kind == 'b' else (np.iinfo(dtype).min, np.iinfo(dtype).max)
        if low < lower or high - 1 > upper: raise RuntimeError('randint bounds exceed the output dtype')
    rng = generator.rng if generator else np.random
    if not 0 < high-low <= 2**32: raise BrowserUnsupported('This randint range requires the server')
    bits = rng.randint(0, 2**32, size=size, dtype=np.uint32).astype(np.uint64)
    return Tensor((bits % (high-low)).astype(dtype) + low)


def full(shape, fill_value, *, dtype=None, device=None, requires_grad=False):
    check_device(device)
    if dtype is None: dtype = np.bool_ if isinstance(fill_value, bool) else np.float32 if isinstance(fill_value, float) else np.int64
    return Tensor(np.full(shape, fill_value, dtype=dtype), requires_grad=requires_grad)


def full_like(input, fill_value, *, dtype=None, device=None, requires_grad=False, **options):
    if options: raise BrowserUnsupported('These full_like options require the server')
    check_device(device)
    return Tensor(np.full_like(input._data,fill_value,dtype=input.dtype if dtype is None else dtype),requires_grad=requires_grad)


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


def cdist(x1, x2, p=2., compute_mode='use_mm_for_euclid_dist_if_necessary'):
    if p != 2 or compute_mode not in ('use_mm_for_euclid_dist_if_necessary', 'use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist'):
        raise BrowserUnsupported('Only Euclidean cdist is supported')
    if x1.ndim < 2 or x2.ndim < 2 or x1.shape[-1] != x2.shape[-1] or x1.dtype != x2.dtype:
        raise RuntimeError('cdist expects matching floating feature dimensions and dtypes')
    if x1.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise BrowserUnsupported('cdist requires float32/float64')
    batch = np.broadcast_shapes(x1.shape[:-2], x2.shape[:-2])
    size = int(np.prod(batch)) * x1.shape[-2] * x2.shape[-2] * max(1, x1.shape[-1])
    if size > 16_000_000: raise BrowserUnsupported('Large pairwise distances require the server')
    # Direct differences avoid the cancellation of ||a||²+||b||²-2a·b near ties.
    difference = x1._data[..., :, None, :] - x2._data[..., None, :, :]
    lengths = np.linalg.norm(difference, axis=-1)
    def direction(g):
        unit = np.divide(difference, lengths[..., None], out=np.zeros_like(difference), where=lengths[..., None] != 0)
        return g[..., None] * unit
    return result(lengths, (x1, lambda g: unbroadcast(direction(g).sum(axis=-2), x1.shape)),
                  (x2, lambda g: unbroadcast(-direction(g).sum(axis=-3), x2.shape)))


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


def meshgrid(*tensors, indexing=None):
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)): tensors = tuple(tensors[0])
    if not tensors or indexing not in (None, 'ij', 'xy'): raise RuntimeError('Invalid meshgrid inputs/indexing')
    if any(x.ndim > 1 or x.dtype != tensors[0].dtype for x in tensors):
        raise RuntimeError('meshgrid expects scalars/vectors with one dtype')
    shape = [x.numel() for x in tensors]
    axes = list(range(len(tensors)))
    if indexing == 'xy' and len(tensors) > 1:
        axes[0], axes[1] = 1, 0
        shape[0], shape[1] = shape[1], shape[0]
    outputs = []
    for x, axis in zip(tensors, axes):
        view = [1] * len(tensors)
        view[axis] = x.numel()
        outputs.append(x.reshape(view).expand(shape))
    return tuple(outputs)


def cartesian_prod(*tensors):
    if not tensors: raise RuntimeError('cartesian_prod expects a non-empty tensor list')
    if any(not isinstance(x, Tensor) or x.ndim != 1 or x.dtype != tensors[0].dtype for x in tensors):
        raise RuntimeError('cartesian_prod expects one-dimensional tensors with the same dtype')
    if len(tensors) == 1: return tensors[0]
    grids = meshgrid(*tensors, indexing='ij')
    return concatenate([grid.reshape(-1) for grid in grids], dim=1, stack=True)


def normalize(input, p=2., dim=1, eps=1e-12, out=None):
    if p != 2 or out is not None: raise BrowserUnsupported('Only Euclidean normalization without out is supported')
    return input / input.norm(2, dim, keepdim=True).clamp_min(eps)


def softplus(input, beta=1., threshold=20.):
    if input.dtype not in (np.dtype('float32'),np.dtype('float64')):
        raise BrowserUnsupported('Softplus requires float32/float64 inputs')
    if not np.isfinite(beta) or beta <= 0 or not np.isfinite(threshold):
        raise BrowserUnsupported('These softplus parameters require the server')
    scaled=beta*input._data
    value=np.where(scaled>threshold,input._data,np.logaddexp(0,scaled)/beta)
    derivative=np.where(scaled>threshold,1,np.exp(-np.logaddexp(0,-scaled)))
    return result(value,(input,lambda g:g*derivative))


def smooth_l1_loss(input, target, size_average=None, reduce=None, reduction='mean', beta=1.):
    """Independent raw NumPy loss and analytical derivatives for both inputs."""
    if size_average is not None or reduce is not None or input.shape != target.shape:
        raise BrowserUnsupported('Legacy or broadcast smooth L1 requires the server')
    if input.dtype != target.dtype or input.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise BrowserUnsupported('Smooth L1 requires matching float32/float64 tensors')
    if not np.isfinite(beta) or beta < 0: raise ValueError('beta must be finite and nonnegative')
    if reduction not in ('none', 'sum', 'mean'): raise ValueError('Invalid smooth L1 reduction')
    difference = input._data - target._data
    absolute = np.abs(difference)
    if beta == 0:
        losses, derivative = absolute, np.sign(difference)
    else:
        quadratic = absolute < beta
        losses = np.where(quadratic, .5 * difference ** 2 / beta, absolute - .5 * beta)
        derivative = np.where(quadratic, difference / beta, np.sign(difference))
    count = difference.size
    value = losses if reduction == 'none' else losses.sum() if reduction == 'sum' else losses.mean() if count else np.asarray(np.nan, dtype=input.dtype)
    def vjp(g): return derivative * g / (max(count, 1) if reduction == 'mean' else 1)
    return result(np.asarray(value, dtype=input.dtype), (input, vjp), (target, lambda g: -vjp(g)))


def mse_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    if size_average is not None or reduce is not None or input.shape != target.shape:
        raise BrowserUnsupported('Legacy or broadcast MSE requires the server')
    if input.dtype != target.dtype or input.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise BrowserUnsupported('MSE requires matching float32/float64 tensors')
    if reduction not in ('none', 'sum', 'mean'): raise ValueError('Invalid MSE reduction')
    losses = (input - target).square()
    return losses if reduction == 'none' else losses.sum() if reduction == 'sum' else losses.mean()


def cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100,
                  reduce=None, reduction='mean', label_smoothing=0.):
    """Independent stable NumPy oracle/VJP for 2D class-index cross entropy."""
    if weight is not None or size_average is not None or reduce is not None or label_smoothing != 0:
        raise BrowserUnsupported('Weighted, smoothed and legacy cross entropy require the server')
    if input.ndim != 2 or target.ndim != 1 or target.dtype != np.dtype('int64'):
        raise BrowserUnsupported('Only [N,C] logits and int64 [N] class targets are supported')
    if input.dtype.kind != 'f' or not np.isfinite(input._data).all():
        raise BrowserUnsupported('Cross entropy requires finite floating logits')
    if target.shape[0] != input.shape[0] or input.shape[1] < 1:
        raise RuntimeError('Invalid cross entropy input/target shapes')
    if reduction not in ('none', 'sum', 'mean'): raise ValueError('Invalid cross entropy reduction')
    labels = target._data
    valid = labels != ignore_index
    if np.any((labels[valid] < 0) | (labels[valid] >= input.shape[1])): raise IndexError('Target out of bounds')
    shifted = input._data - input._data.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    partition = exponentials.sum(axis=1, keepdims=True)
    probabilities = exponentials / partition
    rows = np.flatnonzero(valid)
    losses = np.zeros(input.shape[0], dtype=input.dtype)
    losses[rows] = np.log(partition[rows, 0]) - shifted[rows, labels[rows]]
    count = int(valid.sum())
    value = losses if reduction == 'none' else losses.sum() if reduction == 'sum' else losses.sum() / count if count else np.asarray(np.nan, dtype=input.dtype)
    gradient = probabilities.copy()
    gradient[rows, labels[rows]] -= 1
    gradient[~valid] = 0
    def vjp(g):
        if reduction == 'none': return gradient * np.asarray(g).reshape(-1, 1)
        return gradient * g / (max(count, 1) if reduction == 'mean' else 1)
    return result(np.asarray(value, dtype=input.dtype), (input, vjp))


def unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None):
    input._real_only()
    if input.requires_grad and _grad_enabled: raise BrowserUnsupported('Unique autograd requires the server')
    if dim is not None:
        if not input.ndim or not isinstance(dim,(int,np.integer)) or not -input.ndim <= dim < input.ndim:
            raise IndexError('unique dimension out of range')
        if any(size==0 for size in input.shape): raise BrowserUnsupported('Empty dimension-wise unique requires the server')
    values,inverse,counts=np.unique(input._data,return_inverse=True,return_counts=True,axis=dim,equal_nan=False)
    outputs=[Tensor(values)]
    if return_inverse:outputs.append(Tensor(inverse.reshape(input.shape if dim is None else (input.shape[dim],)),dtype=np.int64))
    if return_counts:outputs.append(Tensor(counts,dtype=np.int64))
    return outputs[0] if len(outputs)==1 else tuple(outputs)


def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2., scale_grad_by_freq=False, sparse=False):
    if max_norm is not None or sparse: raise BrowserUnsupported('Embedding renormalization/sparse gradients require the server')
    if input.dtype not in (np.dtype('int32'),np.dtype('int64')) or weight.ndim!=2:
        raise RuntimeError('Embedding expects integer indices and a matrix codebook')
    if weight.dtype not in (np.dtype('float32'),np.dtype('float64')): raise BrowserUnsupported('Embedding requires real float32/float64 weights')
    if not isinstance(scale_grad_by_freq,bool): raise TypeError('scale_grad_by_freq must be boolean')
    n=weight.shape[0]
    if padding_idx is not None:
        if type(padding_idx) is not int or not -n<=padding_idx<n: raise RuntimeError('Invalid padding index')
        padding_idx%=n
    indices=input._data
    if np.any(indices<0) or np.any(indices>=n): raise IndexError('Embedding index out of bounds')
    def vjp(g):
        gradient=np.zeros_like(weight._data)
        flat=indices.reshape(-1).astype(np.intp,copy=False); incoming=g.reshape(input.numel(),weight.shape[1])
        if scale_grad_by_freq:
            counts=np.bincount(flat,minlength=n)
            incoming=incoming/counts[flat,None]
        if padding_idx is not None:
            valid=flat!=padding_idx;flat=flat[valid];incoming=incoming[valid]
        np.add.at(gradient,flat,incoming)
        return gradient
    return result(weight._data[indices],(weight,vjp),saved=(input,))


def _spatial_pair(value, name, current=None):
    pair=(value,value) if isinstance(value,(int,np.integer)) else tuple(value) if isinstance(value,(tuple,list)) else ()
    if len(pair)!=2: raise BrowserUnsupported(f'{name} requires two spatial dimensions')
    pair=tuple(current[i] if x is None and current is not None else x for i,x in enumerate(pair))
    if any(not isinstance(x,(int,np.integer)) or isinstance(x,(bool,np.bool_)) or x<=0 for x in pair):
        raise RuntimeError(f'{name} requires positive integer sizes')
    return tuple(int(x) for x in pair)


def _resize_matrix(length, output, dtype, *, cubic=False, align=False, factor=None):
    weights=np.zeros((output,length),dtype=dtype)
    if not cubic:
        for i in range(output):
            start=i*length//output;end=((i+1)*length+output-1)//output
            weights[i,start:end]=1/(end-start)
        return weights
    scalar=dtype.type
    scale=scalar((length-1)/(output-1) if align and output>1 else 0 if align else 1/factor if factor is not None else length/output)
    position=np.arange(output,dtype=dtype)*scale if align else (np.arange(output,dtype=dtype)+scalar(.5))*scale-scalar(.5)
    base=np.floor(position).astype(np.int64);t=position-base.astype(dtype)
    a=scalar(-.75)
    def inner(x):return ((a+2)*x-(a+3))*x*x+1
    def outer(x):return ((a*x-5*a)*x+8*a)*x-4*a
    coefficients=(outer(t+1),inner(t),inner(1-t),outer(2-t))
    for offset,coefficient in zip((-1,0,1,2),coefficients):
        np.add.at(weights,(np.arange(output),np.clip(base+offset,0,length-1)),coefficient)
    return weights


def _resize_2d(input, size, *, cubic=False, align=False, factors=None):
    if input.ndim not in (3,4) or min(input.shape[-3:])<=0: raise RuntimeError('2D resize expects nonempty channel and spatial dimensions')
    if input.dtype not in (np.dtype('float32'),np.dtype('float64')): raise BrowserUnsupported('2D resize requires real float32/float64')
    h,w=input.shape[-2:];oh,ow=size
    planes=math.prod(input.shape[:-2])
    if h*oh+w*ow>4_000_000 or planes*max(oh*ow,min(oh*w,h*ow))>8_000_000:
        raise BrowserUnsupported('This resize allocation requires the server')
    wy=_resize_matrix(h,oh,input.dtype,cubic=cubic,align=align,factor=factors[0] if factors else None)
    wx=_resize_matrix(w,ow,input.dtype,cubic=cubic,align=align,factor=factors[1] if factors else None)
    # Choose the smaller intermediate for each direction, including thin maps.
    value=(wy @ input._data) @ wx.T if oh*w<=h*ow else wy @ (input._data @ wx.T)
    def vjp(g):return (wy.T @ g) @ wx if h*ow<=oh*w else wy.T @ (g @ wx)
    return result(value,(input,vjp))


def adaptive_avg_pool2d(input, output_size):
    size=_spatial_pair(output_size,'adaptive_avg_pool2d',input.shape[-2:])
    return _resize_2d(input,size)


def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False):
    if input.ndim!=4 or mode not in ('area','bicubic') or antialias:
        raise BrowserUnsupported('Interpolation supports non-antialiased 2D area/bicubic modes')
    if mode=='area' and align_corners is not None: raise ValueError('align_corners requires an interpolating mode')
    if align_corners is not None and not isinstance(align_corners,bool): raise TypeError('align_corners must be boolean')
    if recompute_scale_factor is not None and not isinstance(recompute_scale_factor,bool): raise TypeError('recompute_scale_factor must be boolean')
    if (size is None)==(scale_factor is None): raise ValueError('Specify size or scale_factor')
    factors=None
    if size is not None:
        if recompute_scale_factor: raise ValueError('recompute_scale_factor requires scale_factor')
        shape=_spatial_pair(size,'interpolate')
    else:
        factors=(scale_factor,scale_factor) if np.isscalar(scale_factor) else tuple(scale_factor)
        if len(factors)!=2 or any(not np.isscalar(x) or not np.isfinite(x) or x<=0 for x in factors): raise ValueError('Invalid scale factors')
        shape=tuple(math.floor(n*f) for n,f in zip(input.shape[-2:],factors))
        shape=_spatial_pair(shape,'interpolate')
        if recompute_scale_factor or mode=='area':factors=None
    return _resize_2d(input,shape,cubic=mode=='bicubic',align=bool(align_corners),factors=factors)


def install():
    torch = types.ModuleType('torch')
    torch.__path__ = []
    torch.Tensor, torch.Generator, torch.no_grad = Tensor, Generator, no_grad
    torch.float64, torch.float32 = np.dtype('float64'), np.dtype('float32')
    torch.complex64, torch.complex128 = np.dtype('complex64'), np.dtype('complex128')
    torch.float, torch.double = torch.float32, torch.float64
    torch.long, torch.int64, torch.bool = np.dtype('int64'), np.dtype('int64'), np.dtype('bool')
    torch.int, torch.int32 = np.dtype('int32'), np.dtype('int32')
    torch.uint8 = np.dtype('uint8')
    torch.inf, torch.pi = np.inf, np.pi
    torch.tensor, torch.where, torch.einsum = tensor, where, einsum
    torch.unique = unique
    torch.remainder = lambda input,other,*,out=None: input.remainder(other,out=out)
    torch.floor_divide = lambda input,other: wrap(input).floor_divide(other)
    torch.trace = lambda input: input.trace()
    torch.det = det
    torch.diagonal = lambda input,offset=0,dim1=0,dim2=1: input.diagonal(offset,dim1,dim2)
    torch.logaddexp = lambda input,other: input.logaddexp(other)
    torch.as_tensor = lambda x: x if isinstance(x, Tensor) else tensor(x)
    torch.from_numpy = lambda x: Tensor(x)
    torch.full, torch.randint, torch.device, torch.finfo = full, randint, Device, np.finfo
    torch.multinomial, torch.randperm = multinomial, randperm
    torch.meshgrid, torch.cartesian_prod = meshgrid, cartesian_prod
    torch.minimum = elementwise_extreme
    torch.maximum = lambda a,b: elementwise_extreme(a,b,maximum=True)
    torch.eye = lambda n,m=None,*,dtype=np.float32,device=None,requires_grad=False: (check_device(device), Tensor(np.eye(n, m, dtype=dtype), requires_grad=requires_grad))[1]
    torch.equal = lambda a,b: bool(np.array_equal(wrap(a)._data, wrap(b)._data))
    torch.cat = concatenate
    torch.stack = lambda tensors,dim=0: concatenate(tensors,dim,stack=True)
    torch.linspace = lambda start,end,steps,dtype=np.float32,device=None,requires_grad=False: (check_device(device), Tensor(np.linspace(start,end,steps,dtype=dtype),requires_grad=requires_grad))[1]
    torch.argsort = lambda x,dim=-1,descending=False,*,stable=False: x.argsort(dim,descending,stable=stable)
    torch.sort = lambda input,dim=-1,descending=False,*,stable=False,out=None: input.sort(dim,descending,stable=stable,out=out)
    torch.trapezoid = trapezoid
    torch.topk = lambda input,k,dim=None,largest=True,sorted=True,*,out=None: input.topk(k,dim,largest,sorted,out=out)
    torch.scatter = lambda input,dim,index,src,*,reduce=None: input.scatter(dim,index,src,reduce=reduce)
    torch.count_nonzero, torch.bincount = count_nonzero, bincount
    torch.manual_seed = _default_generator.manual_seed
    for name in ('randn', 'rand', 'zeros', 'ones', 'empty'):
        setattr(torch, name, lambda *shape, _kind=name, **kw: create(_kind, *shape, **kw))
    for name in ('randn', 'zeros', 'ones', 'empty'):
        setattr(torch, name + '_like', lambda x, _kind=name, **kw: create(_kind, *x.shape, **inherited_options(x.dtype,kw)))
    torch.arange = lambda *args, dtype=None, device=None: (check_device(device), Tensor(np.arange(*args, dtype=dtype or (np.float32 if any(isinstance(x, float) for x in args) else np.int64))))[1]
    torch.full_like = full_like
    torch.unravel_index = lambda index, shape: tuple(Tensor(i) for i in np.unravel_index(int(index), shape))
    torch.broadcast_to = lambda x, shape: x.expand(shape)
    torch.broadcast_shapes = np.broadcast_shapes
    torch.matmul = lambda a,b: wrap(a) @ b
    def outer(input,other):
        if input.ndim!=1 or other.ndim!=1: raise RuntimeError('outer expects vectors')
        return input.unsqueeze(-1)*other.unsqueeze(0)
    torch.outer=outer
    torch.is_floating_point=lambda input: input.is_floating_point()
    def dot(a,b):
        if a.ndim != 1 or b.ndim != 1 or a.dtype != b.dtype: raise RuntimeError('dot expects vectors with matching dtypes')
        return a @ b
    torch.dot, torch.cov = dot, cov
    torch.pdist, torch.cdist = pdist, cdist
    for name in ('sum', 'mean', 'all', 'any', 'softmax', 'exp', 'log', 'sqrt', 'abs', 'relu', 'transpose', 'reshape', 'flatten', 'repeat_interleave', 'max', 'min', 'argmax', 'argmin', 'clamp', 'clamp_min', 'clamp_max', 'sin', 'cos', 'acos', 'erfinv', 'tanh', 'rsqrt', 'sigmoid', 'log_softmax', 'var', 'std', 'cumsum', 'cumprod', 'norm', 'logsumexp', 'diag', 'flip', 'roll', 'triu', 'tril', 'amax', 'prod', 'round', 'chunk'):
        setattr(torch, name, lambda x, *args, _name=name, **kw: getattr(wrap(x), _name)(*args, **kw))
    for name in ('isfinite', 'isnan'):
        setattr(torch, name, lambda x, _name=name: Tensor(getattr(np, _name)(wrap(x)._data)))
    torch.isclose = lambda a,b,**kw: Tensor(np.isclose(wrap(a)._data, wrap(b)._data, **kw))
    torch.allclose = lambda a,b,**kw: bool(torch.isclose(a,b,**kw).all())
    torch.nan_to_num = lambda x,**kw: Tensor(np.nan_to_num(x._data, **kw))
    modules = {}
    torch.logit = lambda input,eps=None: input.logit(eps)
    for name in ('nn', 'nn.functional', 'optim', 'autograd', 'linalg', 'testing', 'random'):
        module = types.ModuleType('torch.' + name)
        def missing(key, _name=name):
            if key.startswith('_'): raise AttributeError(key)
            return unsupported('torch.' + _name + '.' + key)
        module.__getattr__ = missing
        modules[name] = module
        sys.modules['torch.' + name] = module
    torch.nn, torch.optim, torch.autograd, torch.linalg = (modules[k] for k in ('nn', 'optim', 'autograd', 'linalg'))
    torch.nn.functional = modules['nn.functional']
    torch.nn.Module, torch.nn.Parameter = Module, Parameter
    torch.nn.Linear, torch.nn.Sequential, torch.nn.SiLU, torch.nn.ReLU = Linear, Sequential, SiLU, ReLU
    torch.testing = modules['testing']
    torch.testing.assert_close = assert_close
    torch.random = modules['random']
    torch.random.fork_rng = fork_rng
    torch.nn.functional.linear = linear
    torch.nn.functional.embedding = embedding
    torch.nn.functional.interpolate = interpolate
    torch.nn.functional.adaptive_avg_pool2d = adaptive_avg_pool2d
    torch.nn.functional.softmax = lambda x,dim,dtype=None: x.softmax(dim, dtype)
    torch.nn.functional.log_softmax = lambda x,dim,dtype=None: x.log_softmax(dim,dtype)
    torch.nn.functional.logsigmoid = lambda x: x.logsigmoid()
    torch.nn.functional.normalize, torch.nn.functional.cross_entropy = normalize, cross_entropy
    torch.nn.functional.smooth_l1_loss = smooth_l1_loss
    torch.nn.functional.mse_loss = mse_loss
    torch.nn.functional.softplus = softplus
    torch.nn.functional.silu = silu
    torch.nn.functional.layer_norm = layer_norm_oracle
    torch.nn.functional.rms_norm = rms_norm_oracle
    torch.nn.functional.scaled_dot_product_attention = scaled_dot_product_attention_oracle
    def relu(x, inplace=False):
        if inplace: raise BrowserUnsupported('In-place ReLU requires the server')
        return x.relu()
    torch.nn.functional.relu = relu
    torch.optim.Adam, torch.optim.SGD, torch.autograd.grad = Adam, SGD, autograd_grad
    torch.autograd.gradcheck = gradcheck
    torch.optim.AdamW = AdamW
    torch.linalg.matrix_rank = lambda x: Tensor(np.linalg.matrix_rank(x._data), dtype=np.int64)
    torch.linalg.solve = solve
    torch.linalg.inv = inv
    torch.linalg.norm = linalg_norm
    torch.linalg.svd, torch.linalg.eigh, torch.linalg.qr = svd, eigh, qr
    torch.linalg.eigvalsh, torch.linalg.det = eigvalsh, det
    torch.linalg.svdvals = lambda input,*,driver=None,out=None: svd(input,False,driver=driver,out=out).S
    torch.__getattr__ = unsupported
    return torch

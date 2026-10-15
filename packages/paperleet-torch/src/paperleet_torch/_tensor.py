"""NumPy-backed tensor implementation; extracted from PaperLeet."""
import copy
import math
import weakref
from collections import namedtuple
from statistics import NormalDist
import numpy as np


class StorageVersion:
    __slots__ = ('version', '_linked', '__weakref__')

    def __init__(self):
        self.version = 0
        self._linked = None

    def bump(self):
        self.version += 1
        if self._linked is not None:
            if len(self._linked) == 1:
                self._linked = None
                return
            for counter in self._linked:
                if counter is not self:
                    counter.version += 1

    def link(self, other):
        """Keep existing graph versions while joining rebinding mutation guards."""
        if self is other or self._linked is not None and self._linked is other._linked:
            return
        members = self._linked if self._linked is not None else weakref.WeakSet([self])
        incoming = list(other._linked) if other._linked is not None else [other]
        members.update(incoming)
        for counter in members:
            counter._linked = members


def storage_version(data):
    """Views, detach and dtype-preserving casts share their NumPy allocation's counter."""
    owner = data
    while isinstance(owner.base, np.ndarray): owner = owner.base
    key = id(owner)
    previous = _state._storages.get(key)
    if previous is not None and previous[0]() is owner: return previous[1]
    counter = StorageVersion()
    def release(reference):
        if _state._storages.get(key, (None,))[0] is reference: _state._storages.pop(key, None)
    _state._storages[key] = (weakref.ref(owner, release), counter)
    return counter


class Tensor:
    __array_priority__ = 1000
    __hash__ = object.__hash__
    _joint_vjp = None

    def __init__(self, data, dtype=None, requires_grad=False):
        self._data = np.asarray(data, dtype=dtype)
        self._storage = storage_version(self._data)
        self._saved_versions = ()
        self.requires_grad = bool(requires_grad)
        if requires_grad and self.dtype.kind == 'c': raise _errors.BrowserUnsupported('Complex autograd requires the server')
        if requires_grad and self.dtype.kind != 'f':
            raise RuntimeError('Only floating point tensors can require gradients')
        self._parents = ()
        self.grad = None

    shape = property(lambda self: self._data.shape)
    ndim = property(lambda self: self._data.ndim)
    dtype = property(lambda self: self._data.dtype)
    device = property(lambda self: _device._CPU)
    is_leaf = property(lambda self: not self._parents)
    _version = property(lambda self: self._storage.version)
    T = property(lambda self: self.permute(*reversed(range(self.ndim))))
    data = property(lambda self: self.detach())

    def __getattr__(self, name):
        if name.startswith('_'): raise AttributeError(name)
        return _errors.unsupported(f'Tensor.{name}')
    def _real_only(self):
        if self.dtype.kind == 'c': raise _errors.BrowserUnsupported('This complex operation requires the server')
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
    def count_nonzero(self, dim=None): return _ops.count_nonzero(self, dim)
    def size(self, dim=None): return self.shape if dim is None else self.shape[dim]
    def detach(self): return Tensor(self._data)
    def clone(self, *, memory_format=None):
        fmt = _conversion.memory_format(memory_format)
        data = self._data.copy(order='C' if fmt is _conversion.MemoryFormat.contiguous_format else 'K')
        return _autograd.result(data, (self, lambda g: g))
    def cpu(self, memory_format=None):
        if memory_format is None or memory_format is _conversion.MemoryFormat.preserve_format:
            return self
        return self.to(device='cpu', memory_format=memory_format)
    def contiguous(self, memory_format=None):
        _conversion.memory_format(memory_format, contiguous=True)
        if self._data.flags.c_contiguous:
            return self
        return self.clone(memory_format=_conversion.MemoryFormat.contiguous_format)
    def is_contiguous(self, memory_format=None):
        _conversion.memory_format(memory_format)
        return bool(self._data.flags.c_contiguous)
    def double(self): return self.to(dtype=np.float64)
    def float(self): return self.to(dtype=np.float32)
    def long(self): return self.to(dtype=np.int64)
    def int(self): return self.to(dtype=np.int32)
    def bool(self): return self.to(dtype=np.bool_)
    def new_zeros(self, *shape, **kw): return _random.create('zeros', *shape, **_ops.inherited_options(self.dtype,kw))
    def new_ones(self, *shape, **kw): return _random.create('ones', *shape, **_ops.inherited_options(self.dtype,kw))
    def new_empty(self, *shape, **kw): return _random.create('empty', *shape, **_ops.inherited_options(self.dtype,kw))
    def new_full(self, size, fill_value, **kw): return _ops.full(size, fill_value, **_ops.inherited_options(self.dtype,kw))
    def new_tensor(self, data, *, dtype=None, device=None, requires_grad=False):
        _device.check_device(device)
        return Tensor(np.array(data._data if isinstance(data, Tensor) else data, dtype=dtype or self.dtype, copy=True), requires_grad=requires_grad)
    def zero_(self): self[...] = 0; return self
    def type_as(self, other): return self.to(other)

    def to(self, *args, **kwargs):
        if not args:
            if not kwargs:
                return self
            if len(kwargs) == 1:
                if 'dtype' in kwargs and (kwargs['dtype'] is None or kwargs['dtype'] is self.dtype or kwargs['dtype'] is self.dtype.type):
                    return self
                if kwargs.get('device') == 'cpu':
                    return self
        elif len(args) == 1 and not kwargs:
            if args[0] is self.dtype or isinstance(args[0], str) and args[0] == 'cpu':
                return self
        _, dtype, _, copy, fmt = _conversion.parse_to(args, kwargs)
        target = self.dtype if dtype is None else dtype
        if self.dtype.kind == 'c' and target.kind != 'c':
            raise _errors.BrowserUnsupported('Casting complex tensors to real values requires the server')
        contiguous = fmt is _conversion.MemoryFormat.contiguous_format
        if target == self.dtype and not copy:
            return self
        data = self._data.astype(target, order='C' if contiguous else 'K', copy=copy)
        return _autograd.result(data, (self, lambda g: g)) if data.dtype.kind in 'fc' else Tensor(data)

    def type(self, dtype=None, non_blocking=False, **kwargs):
        if dtype is None:
            name = {'float32': 'FloatTensor', 'float64': 'DoubleTensor', 'int64': 'LongTensor',
                    'int32': 'IntTensor', 'uint8': 'ByteTensor', 'bool': 'BoolTensor',
                    'complex64': 'ComplexFloatTensor', 'complex128': 'ComplexDoubleTensor'}.get(self.dtype.name)
            if name is None: raise _errors.BrowserUnsupported('Tensor type name is unsupported for this dtype')
            return 'torch.' + name
        return self.to(dtype=_conversion.type_dtype(dtype), non_blocking=non_blocking, **kwargs)

    def _replace_data(self, value):
        if _state._jvp_active:
            raise _errors.BrowserUnsupported('Rebinding module storage during JVP is unsupported')
        if not isinstance(value, Tensor):
            raise TypeError('Module conversion must return a Tensor')
        if value is self:
            return
        if value.shape != self.shape:
            raise _errors.BrowserUnsupported('Shape-changing module storage conversion is unsupported')
        if self._parents:
            raise _errors.BrowserUnsupported('Rebinding non-leaf gradient storage is unsupported')
        if self.requires_grad and value.dtype.kind != 'f':
            if value.dtype.kind == 'c':
                raise _errors.BrowserUnsupported('Complex module autograd is unsupported')
            raise RuntimeError('Differentiable module data must be floating point')
        self._storage.link(value._storage)
        self._data = value._data
        # Future views use the retained parameter counter. The temporary cast
        # counter can then disappear; externally held cast tensors stay linked.
        owner = self._data
        while isinstance(owner.base, np.ndarray): owner = owner.base
        previous = _state._storages.get(id(owner))
        if previous is not None and previous[0]() is owner:
            _state._storages[id(owner)] = (previous[0], self._storage)

    def requires_grad_(self, requires_grad=True):
        if _state._jvp_active: raise _errors.BrowserUnsupported('Changing gradient flags inside JVP requires the server')
        if requires_grad and self.dtype.kind == 'c': raise _errors.BrowserUnsupported('Complex autograd requires the server')
        if requires_grad and self.dtype.kind != 'f': raise RuntimeError('Only floating tensors can require grad')
        self.requires_grad = requires_grad
        return self

    def backward(self, gradient=None, *, retain_graph=False, create_graph=False):
        if create_graph: raise _errors.BrowserUnsupported('Higher-order autograd is not supported')
        if gradient is None and self.numel() != 1: raise RuntimeError('Non-scalar outputs require a gradient')
        seed = np.ones_like(self._data) if gradient is None else _ops.wrap(gradient)._data
        order, gradients = _autograd.backprop(self, seed)
        for node in order:
            if node.requires_grad and not node._parents and id(node) in gradients:
                grad = gradients[id(node)]
                if node.grad is None:
                    node.grad = Tensor(grad.copy())
                else:
                    if not isinstance(node.grad, Tensor) or node.grad.shape != grad.shape or node.grad.dtype.kind != 'f':
                        raise RuntimeError('Existing gradient must be a floating Tensor of the correct shape')
                    node.grad._data[...] += grad
                    node.grad._storage.bump()

    def __getitem__(self, index):
        original_data = self._data
        saved = []
        def convert(key):
            if isinstance(key, Tensor): saved.append(key); return key._data
            if isinstance(key, tuple): return tuple(convert(k) for k in key)
            return key
        index = convert(index)
        keys = index if isinstance(index, tuple) else (index,)
        basic = all(key is None or key is Ellipsis or isinstance(key, slice) or
                    isinstance(key, (int, np.integer)) and not isinstance(key, (bool, np.bool_))
                    for key in keys)
        def vjp(g):
            grad = np.zeros_like(original_data)
            # Basic slices address each destination once. Advanced indexing can
            # repeat destinations, and must retain unbuffered scatter addition.
            if basic:
                grad[index] += g
            else:
                np.add.at(grad, index, g)
            return grad
        return _autograd.result(original_data[index], (self, vjp), saved=saved)

    def __setitem__(self, index, value):
        if _state._jvp_active: raise _errors.BrowserUnsupported('Tensor mutation during JVP requires the server')
        if _state._grad_enabled and (self.requires_grad or isinstance(value, Tensor) and value.requires_grad):
            raise _errors.BrowserUnsupported('In-place edits of differentiable tensors require the server')
        if self.requires_grad and self._parents:
            raise _errors.BrowserUnsupported('In-place updates of non-leaf tensors require the server')
        def convert(key):
            if isinstance(key, Tensor): return key._data
            if isinstance(key, tuple): return tuple(convert(k) for k in key)
            return key
        index = convert(index)
        self._data[index] = value._data if isinstance(value, Tensor) else value
        self._storage.bump()

    def fill_(self, value):
        if isinstance(value, Tensor) and value.ndim:
            raise RuntimeError('fill_ requires a scalar value')
        self[...] = value
        return self

    def __deepcopy__(self, memo):
        if self._parents:
            raise _errors.BrowserUnsupported('Copying non-leaf autograd tensors requires the server')
        owner = self._data
        while isinstance(owner.base, np.ndarray): owner = owner.base
        if owner.dtype.hasobject or not (owner.flags.c_contiguous or owner.flags.f_contiguous):
            raise _errors.BrowserUnsupported('Copying tensors with non-dense backing storage is unsupported')
        # Memoize the allocation, not each view: detached checkpoint entries can
        # share storage with different shapes, offsets and strides.
        cloned_owner = copy.deepcopy(owner, memo)
        offset = self._data.__array_interface__['data'][0] - owner.__array_interface__['data'][0]
        data = np.ndarray(self.shape, dtype=self.dtype, buffer=cloned_owner,
                          offset=offset, strides=self._data.strides)
        value = type(self)(data, requires_grad=self.requires_grad)
        memo[id(self)] = value
        value.grad = copy.deepcopy(self.grad, memo)
        return value

    def _binary(self, other, op, left, right):
        if not isinstance(other, (Tensor, np.ndarray)):
            if isinstance(other, (complex, np.complexfloating)):
                other = Tensor(other, dtype=np.complex128 if self.dtype in (np.dtype('float64'),np.dtype('complex128')) else np.complex64)
            elif self.dtype.kind in 'fc': other = Tensor(other, dtype=self.dtype)
        other = _ops.wrap(other)
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
        return _autograd.result(op(a, b), (self, lambda g: _ops.unbroadcast(left(g, a, b), a.shape)),
                      (other, lambda g: _ops.unbroadcast(right(g, a, b), b.shape)))
    def __add__(self, other): return self._binary(other, np.add, lambda g,a,b:g, lambda g,a,b:g)
    __radd__ = __add__
    def __sub__(self, other): return self._binary(other, np.subtract, lambda g,a,b:g, lambda g,a,b:-g)
    def __rsub__(self, other): return -self + other
    def __mul__(self, other): return self._binary(other, np.multiply, lambda g,a,b:g*b, lambda g,a,b:g*a)
    __rmul__ = __mul__
    def __truediv__(self, other): return self._binary(other, np.divide, lambda g,a,b:g/b, lambda g,a,b:-g*a/(b*b))
    def __rtruediv__(self, other): return self._binary(other, lambda a,b: b/a, lambda g,a,b: -g*b/(a*a), lambda g,a,b: g/a)
    def floor_divide(self, other):
        other_value = _ops.wrap(other)
        self._real_only(); other_value._real_only()
        if self.dtype.kind == other_value.dtype.kind == 'b': raise _errors.BrowserUnsupported('Boolean floor division requires the server')
        if _state._grad_enabled and (self.requires_grad or other_value.requires_grad):
            raise _errors.BrowserUnsupported('Floor division has no supported backward; use the server')
        if self.dtype.kind in 'biu' and other_value.dtype.kind in 'biu':
            if np.any(other_value._data == 0): raise RuntimeError('Integer division by zero')
            if self.dtype.kind == 'i' and np.any(self._data == np.iinfo(self.dtype).min) and np.any(other_value._data == -1):
                raise _errors.BrowserUnsupported('Overflowing integer floor division requires the server')
        with np.errstate(divide='ignore', invalid='ignore'):
            return self._binary(other, np.floor_divide, lambda *args: _errors.unsupported('floor_divide backward'), lambda *args: _errors.unsupported('floor_divide backward'))
    __floordiv__ = floor_divide
    def __rfloordiv__(self, other): return _ops.wrap(other).floor_divide(self)
    def remainder(self, other, *, out=None):
        if out is not None: raise _errors.BrowserUnsupported('Remainder output buffers require the server')
        self._real_only(); _ops.wrap(other)._real_only()
        if self.dtype.kind=='b' and _ops.wrap(other).dtype.kind=='b': raise _errors.BrowserUnsupported('Boolean remainder requires the server')
        if self.dtype.kind in 'biu' and _ops.wrap(other).dtype.kind in 'biu' and np.any(_ops.wrap(other)._data==0):
            raise RuntimeError('Integer division by zero')
        return self._binary(other,np.remainder,lambda g,a,b:g,lambda g,a,b:-g*np.floor(a/b))
    __mod__ = remainder
    def __rmod__(self, other):
        self._real_only(); _ops.wrap(other)._real_only()
        if self.dtype.kind=='b' and _ops.wrap(other).dtype.kind=='b': raise _errors.BrowserUnsupported('Boolean remainder requires the server')
        if self.dtype.kind in 'biu' and _ops.wrap(other).dtype.kind in 'biu' and np.any(self._data==0):
            raise RuntimeError('Integer division by zero')
        return self._binary(other,lambda a,b:np.remainder(b,a),lambda g,a,b:-g*np.floor(b/a),lambda g,a,b:g)
    def __neg__(self): return _autograd.result(-self._data, (self, lambda g: -g))
    def _inplace(self, other, operation):
        if _state._grad_enabled and (self.requires_grad or _ops.wrap(other).requires_grad):
            raise _errors.BrowserUnsupported('In-place updates of differentiable tensors require the server')
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
        original_data = self._data
        if not np.isscalar(exponent): raise _errors.BrowserUnsupported('Tensor exponents require the server')
        if exponent == 0:
            derivative = lambda g: np.zeros_like(g)
        elif exponent == 2:
            # Keep native saved-self behavior without materializing data ** 1.
            derivative = lambda g: g * (2 * self._data)
        else:
            derivative = lambda g: g * (exponent * self._data ** (exponent - 1))
        return _autograd.result(original_data ** exponent, (self, derivative))
    def __rpow__(self, base):
        if type(base) not in (int, float) or not math.isfinite(base) or base <= 0 or self.dtype not in (np.dtype('float32'), np.dtype('float64')):
            raise _errors.BrowserUnsupported('Scalar-base power requires a positive finite real base and float32/float64 exponents')
        with np.errstate(over='ignore', under='ignore'):
            converted = np.asarray(base, dtype=self.dtype)
        if not np.isfinite(converted) or converted <= 0:
            raise _errors.BrowserUnsupported('Power base is outside the exponent dtype range')
        with np.errstate(over='ignore', invalid='ignore'):
            value = np.power(converted, self._data)
        # Native scalar power retains the original scalar in log(base), even
        # when float32 rounds the forward base to exactly one.
        return _autograd.result(value, (self, lambda g: g * value * math.log(base)))
    def pow(self, exponent): return self ** exponent
    def square(self): return self ** 2
    def sqrt(self): return self ** .5
    def abs(self):
        original_data = self._data
        return _autograd.result(np.abs(original_data), (self, lambda g: g * np.sign(self._data)))
    def relu(self):
        original_data = self._data
        self._real_only()
        return _autograd.result(np.maximum(original_data, 0), (self, lambda g: np.where(original_data <= 0, 0, g)))
    def exp(self):
        data = np.exp(self._data)
        return _autograd.result(data, (self, lambda g: g * data))
    def log(self):
        original_data = self._data
        return _autograd.result(np.log(original_data), (self, lambda g: g / self._data))
    def logit(self, eps=None):
        original_data = self._data
        if original_data.dtype.kind != 'f': raise _errors.BrowserUnsupported('logit requires floating tensors')
        if eps is not None and (not np.isfinite(eps) or not 0 <= eps <= .5):
            raise _errors.BrowserUnsupported('logit eps outside [0, .5] requires the server')
        probability = original_data if eps is None else np.clip(original_data, eps, 1 - eps)
        with np.errstate(divide='ignore', invalid='ignore'):
            value = np.log(probability / (1 - probability))
        def vjp(g):
            source = self._data.astype(np.result_type(g.dtype, self.dtype), copy=False)
            with np.errstate(divide='ignore', invalid='ignore'):
                gradient = g / (source * (1 - source))
            if eps is None:
                gradient = np.where((source < 0) | (source > 1), np.nan, gradient)
            else:
                gradient = np.where((source < eps) | (source > 1 - eps), 0., gradient)
            return gradient
        return _autograd.result(value, (self, vjp))
    def logaddexp(self, other):
        self._real_only(); _ops.wrap(other)._real_only()
        return self._binary(other, np.logaddexp,
                            lambda g,a,b: g * np.exp(-np.logaddexp(0, b-a)),
                            lambda g,a,b: g * np.exp(-np.logaddexp(0, a-b)))

    def __matmul__(self, other):
        other = _ops.wrap(other)
        a, b = self._data, other._data
        if a.dtype != b.dtype: raise RuntimeError('matmul operands must have the same dtype')
        # Promote vectors to matrices before calculating batched VJPs.
        aa, bb = a[None, :] if a.ndim == 1 else a, b[:, None] if b.ndim == 1 else b
        def promote(g):
            if b.ndim == 1: g = np.expand_dims(g, -1)
            if a.ndim == 1: g = np.expand_dims(g, -2)
            return g
        def left(g):
            if a.ndim == b.ndim == 1:
                return g.astype(other.dtype, copy=False) * other._data
            if a.ndim >= 2 and b.ndim == 1:
                return g[..., None] * other._data
            current = other._data if b.ndim == 2 else b
            current = current[:, None] if b.ndim == 1 else current
            if g.dtype != current.dtype:
                raise RuntimeError('matmul backward operands must have matching dtypes')
            grad = _ops.unbroadcast(promote(g) @ np.swapaxes(current, -1, -2), aa.shape)
            return grad.reshape(a.shape)
        def right(g):
            if a.ndim == b.ndim == 1:
                return g.astype(self.dtype, copy=False) * self._data
            if a.ndim == 1 and b.ndim >= 3:
                return self._data[:, None] * g[..., None, :]
            current = self._data if a.ndim == 2 and b.ndim <= 2 else a
            current = current[None, :] if a.ndim == 1 else current
            if current.dtype != g.dtype:
                raise RuntimeError('matmul backward operands must have matching dtypes')
            grad = _ops.unbroadcast(np.swapaxes(current, -1, -2) @ promote(g), bb.shape)
            return grad.reshape(b.shape)
        return _autograd.result(a @ b, (self, left), (other, right))
    def __rmatmul__(self, other): return _ops.wrap(other) @ self

    def reshape(self, *shape):
        original_data = self._data
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)): shape = tuple(shape[0])
        return _autograd.result(original_data.reshape(shape), (self, lambda g: g.reshape(original_data.shape)))
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
    def unsqueeze(self, dim):
        original_data = self._data
        return _autograd.result(np.expand_dims(original_data, dim), (self, lambda g: g.reshape(original_data.shape)))
    def squeeze(self, dim=None):
        original_data = self._data
        if dim is not None and original_data.shape[dim] != 1: return self
        return _autograd.result(np.squeeze(original_data, axis=dim), (self, lambda g: g.reshape(original_data.shape)))
    def permute(self, *dims):
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)): dims = tuple(dims[0])
        dims = tuple(d % self.ndim for d in dims)
        return _autograd.result(self._data.transpose(dims), (self, lambda g: g.transpose(np.argsort(dims))))
    def transpose(self, dim0, dim1):
        dims = list(range(self.ndim)); dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
        return self.permute(*dims)
    def expand(self, *shape):
        original_data = self._data
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)): shape = tuple(shape[0])
        leading = (1,) * (len(shape) - original_data.ndim) + original_data.shape
        target = tuple(old if new == -1 else new for old, new in zip(leading, shape))
        return _autograd.result(np.broadcast_to(original_data, target), (self, lambda g: _ops.unbroadcast(g, original_data.shape)))
    def expand_as(self, other): return self.expand(other.shape)
    def repeat(self, *repeats):
        original_data = self._data
        if len(repeats) == 1 and isinstance(repeats[0], (tuple, list)): repeats = tuple(repeats[0])
        shape = (1,) * (len(repeats) - original_data.ndim) + original_data.shape
        interleaved = tuple(v for pair in zip(repeats, shape) for v in pair)
        return _autograd.result(np.tile(original_data, repeats), (self, lambda g: g.reshape(interleaved).sum(axis=tuple(range(0, len(interleaved), 2))).reshape(original_data.shape)))
    def repeat_interleave(self, repeats, dim=None, *, output_size=None):
        source = self.flatten() if dim is None else self
        dim = 0 if dim is None else dim
        if not -source.ndim <= dim < source.ndim: raise IndexError('repeat_interleave dimension out of range')
        dim %= source.ndim
        counts = _ops.wrap(repeats)._data
        if counts.dtype.kind not in 'iu' or counts.ndim > 1 or np.any(counts < 0): raise RuntimeError('Invalid repetition counts')
        if np.any(counts > np.iinfo(np.intp).max):
            raise _errors.BrowserUnsupported('Repetition counts exceed this runtime index range')
        # wasm32 uses int32 indices even though torch.long remains int64.
        indices = np.repeat(np.arange(source.shape[dim], dtype=np.intp), counts.astype(np.intp))
        if output_size is not None and output_size != len(indices): raise RuntimeError('Incorrect repeat_interleave output_size')
        index = [slice(None)] * source.ndim; index[dim] = indices
        return source[tuple(index)]

    def sum(self, dim=None, keepdim=False, *, dtype=None):
        original_data = self._data
        dims = tuple(range(original_data.ndim)) if dim is None else tuple(d % original_data.ndim for d in (dim if isinstance(dim, (list, tuple)) else (dim,)))
        def vjp(g):
            if not keepdim:
                for axis in sorted(dims): g = np.expand_dims(g, axis)
            return np.broadcast_to(g, original_data.shape)
        if dtype is None and original_data.dtype.kind in 'biu': dtype = np.int64
        return _autograd.result(original_data.sum(axis=dims, keepdims=keepdim, dtype=dtype), (self, vjp))
    def mean(self, dim=None, keepdim=False):
        dims = tuple(range(self.ndim)) if dim is None else dim if isinstance(dim, (list, tuple)) else (dim,)
        return self.sum(dim, keepdim) / int(np.prod([self.shape[d] for d in dims]))
    def trace(self):
        original_data = self._data
        if original_data.ndim != 2: raise RuntimeError('trace expects a matrix')
        self._real_only()
        return _autograd.result(np.trace(original_data), (self, lambda g: np.eye(*original_data.shape, dtype=original_data.dtype) * g))
    def det(self): return _linalg.det(self)
    def prod(self, dim=None, keepdim=False, *, dtype=None):
        original_data = self._data
        if dim is None and keepdim: raise TypeError('prod keepdim requires an explicit dimension')
        if dim is not None and not isinstance(dim,(int,np.integer)): raise _errors.BrowserUnsupported('prod requires one dimension')
        axes=_ops.reduction_axes(dim,original_data.ndim)
        if dtype is not None and np.dtype(dtype) != original_data.dtype: return self.to(dtype=dtype).prod(dim,keepdim)
        target = np.int64 if dtype is None and original_data.dtype.kind in 'biu' else dtype
        value=np.prod(original_data,axis=axes,keepdims=True,dtype=target)
        zeros=original_data==0
        count=zeros.sum(axis=axes,keepdims=True)
        nonzero=np.prod(np.where(zeros,1,original_data),axis=axes,keepdims=True,dtype=target)
        def vjp(g):
            source = self._data
            if not np.isfinite(source).all(): raise _errors.BrowserUnsupported('Nonfinite product gradients require the server')
            if not keepdim:
                for axis in sorted(axes): g=np.expand_dims(g,axis)
            active_zeros, active_count, active_nonzero = zeros, count, nonzero
            if source is not original_data:
                active_zeros = source == 0
                active_count = active_zeros.sum(axis=axes, keepdims=True)
                active_nonzero = np.prod(np.where(active_zeros, 1, source), axis=axes, keepdims=True)
            common = np.result_type(value.dtype, source.dtype, g.dtype)
            ordinary=np.divide(value,source,out=np.zeros(source.shape,dtype=common),where=~active_zeros)
            derivative=np.where(active_count==0,ordinary,np.where((active_count==1)&active_zeros,active_nonzero,0))
            return g*derivative
        return _autograd.result(value if keepdim else value.squeeze(axis=axes),(self,vjp))
    def amax(self, dim=None, keepdim=False, *, out=None):
        self._real_only()
        if out is not None: raise _errors.BrowserUnsupported('amax output buffers require the server')
        axes=_ops.reduction_axes(dim,self.ndim)
        source = self._data
        value=np.max(source,axis=axes,keepdims=True)
        mask=source==value
        def vjp(g):
            active_mask = mask
            if self._data is not source:
                saved = value if keepdim else value.squeeze(axis=axes)
                if not keepdim and saved.ndim:
                    for axis in sorted(axes): saved = np.expand_dims(saved, axis)
                active_mask = (Tensor(self._data) == Tensor(saved))._data
            if not keepdim:
                for axis in sorted(axes): g=np.expand_dims(g,axis)
            with np.errstate(divide='ignore',invalid='ignore'):
                return g*active_mask/active_mask.sum(axis=axes,keepdims=True)
        return _autograd.result(value if keepdim else value.squeeze(axis=axes),(self,vjp))
    def round(self, *, decimals=0, out=None):
        original_data = self._data
        if decimals!=0 or out is not None: raise _errors.BrowserUnsupported('Decimal round/output buffers require the server')
        if original_data.dtype.kind in 'iu': return self.clone()
        if original_data.dtype.kind!='f': raise _errors.BrowserUnsupported('round requires real floating or integer inputs')
        return _autograd.result(np.rint(original_data),(self,lambda g: np.zeros_like(original_data)))
    def norm(self, p='fro', dim=None, keepdim=False, dtype=None, *, out=None):
        if out is not None: raise _errors.BrowserUnsupported('Norm out arguments require the server')
        if p not in (2, 'fro'): raise _errors.BrowserUnsupported('Only Euclidean/Frobenius norms are supported')
        if dtype is not None: return self.to(dtype=dtype).norm(p, dim, keepdim)
        if self.dtype.kind != 'f': raise _errors.BrowserUnsupported('Norms require a floating tensor')
        axes = _ops.reduction_axes(dim, self.ndim)
        if p == 'fro' and dim is not None and len(axes) > 2:
            raise RuntimeError('Frobenius norm accepts at most two dimensions')
        source = self._data
        lengths = np.sqrt(np.sum(source ** 2, axis=axes, keepdims=True))
        direction = np.divide(source, lengths, out=np.zeros_like(source), where=lengths != 0)
        def vjp(g):
            if self._data is not source:
                saved = lengths if keepdim else lengths.squeeze(axis=axes)
                with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                    alpha = g / saved
                    mask = saved == 0
                    if not keepdim and alpha.ndim:
                        for axis in sorted(axes):
                            alpha, mask = np.expand_dims(alpha, axis), np.expand_dims(mask, axis)
                    value = (Tensor(self._data) * Tensor(alpha))._data
                return np.where(mask, 0., value)
            if not keepdim:
                for axis in sorted(axes): g = np.expand_dims(g, axis)
            return g * direction
        return _autograd.result(lengths if keepdim else lengths.squeeze(axis=axes), (self, vjp))
    def logsumexp(self, dim, keepdim=False, *, out=None):
        if out is not None: raise _errors.BrowserUnsupported('logsumexp out arguments require the server')
        if dim is None: raise TypeError('logsumexp requires a dimension')
        if self.dtype.kind != 'f': raise _errors.BrowserUnsupported('logsumexp requires a floating tensor')
        axes = _ops.reduction_axes(dim, self.ndim)
        maximum = np.max(self._data, axis=axes, keepdims=True, initial=-np.inf)
        shift = np.where(np.isfinite(maximum), maximum, 0)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            value = np.log(np.exp(self._data - shift).sum(axis=axes, keepdims=True)) + shift
            derivative = np.exp(self._data - value)
        def vjp(g):
            if not keepdim:
                for axis in sorted(axes): g = np.expand_dims(g, axis)
            return g * derivative
        return _autograd.result(value if keepdim else value.squeeze(axis=axes), (self, vjp))
    def diagonal(self, offset=0, dim1=0, dim2=1):
        original_data = self._data
        if original_data.ndim < 2: raise RuntimeError('diagonal requires at least two dimensions')
        if not isinstance(offset, (int, np.integer)): raise TypeError('Diagonal offset must be an integer')
        axes = _ops.reduction_axes((dim1, dim2), original_data.ndim)
        value = np.diagonal(original_data, offset=int(offset), axis1=axes[0], axis2=axes[1])
        if original_data.flags.writeable: value.setflags(write=True)
        def vjp(g):
            gradient = np.zeros_like(original_data)
            moved = np.moveaxis(gradient, axes, (-2, -1))
            index = np.arange(value.shape[-1])
            moved[..., index + max(-int(offset), 0), index + max(int(offset), 0)] = g
            return gradient
        return _autograd.result(value, (self, vjp))
    def diag(self, diagonal=0, *, out=None):
        original_data = self._data
        if out is not None: raise _errors.BrowserUnsupported('diag out arguments require the server')
        if original_data.ndim not in (1, 2): raise RuntimeError('diag expects a vector or matrix')
        value = np.diag(original_data, k=diagonal)
        def vjp(g):
            if original_data.ndim == 1: return np.diag(g, k=diagonal)
            gradient = np.zeros_like(original_data)
            positions = np.arange(value.size, dtype=np.intp)
            gradient[positions + max(-diagonal, 0), positions + max(diagonal, 0)] = g
            return gradient
        return _autograd.result(value, (self, vjp))
    def any(self, dim=None, keepdim=False): return Tensor(self._data.any(axis=dim, keepdims=keepdim))
    def all(self, dim=None, keepdim=False): return Tensor(self._data.all(axis=dim, keepdims=keepdim))
    def _extreme(self, dim, keepdim, maximum):
        original_data = self._data
        self._real_only()
        op, argop = (np.max, np.argmax) if maximum else (np.min, np.argmin)
        data = op(original_data, axis=dim, keepdims=keepdim)
        if dim is None:
            mask = original_data == data
            return _autograd.result(data, (self, lambda g: g * mask / mask.sum()))
        indices = argop(original_data, axis=dim)
        def vjp(g):
            grad = np.zeros_like(original_data)
            np.put_along_axis(grad, np.expand_dims(indices, dim), g if keepdim else np.expand_dims(g, dim), axis=dim)
            return grad
        pair = namedtuple('Reduction', ['values', 'indices'])
        return pair(_autograd.result(data, (self, vjp)), Tensor(np.expand_dims(indices, dim) if keepdim else indices))
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
        if self.dtype.kind not in 'fiub': raise _errors.BrowserUnsupported('Sorting this dtype requires the server')
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
        if out is not None: raise _errors.BrowserUnsupported('Sort output buffers require the server')
        indices = self.argsort(dim, descending, stable=stable)
        pair = namedtuple('Sort', ['values', 'indices'])
        return pair(self.gather(dim, indices) if self.ndim else self.clone(), indices)
    def topk(self, k, dim=None, largest=True, sorted=True, *, out=None):
        if out is not None: raise _errors.BrowserUnsupported('topk output buffers require the server')
        if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)): raise TypeError('topk k must be an integer')
        if not isinstance(largest, bool) or not isinstance(sorted, bool): raise TypeError('topk flags must be boolean')
        dim = -1 if dim is None else dim
        if not isinstance(dim, (int, np.integer)) or not -max(self.ndim, 1) <= dim < max(self.ndim, 1):
            raise IndexError('topk dimension out of range')
        if self.dtype.kind not in 'fiu': raise _errors.BrowserUnsupported('topk for this dtype requires the server')
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
        if reduce is not None: raise _errors.BrowserUnsupported('Scatter reductions require the server')
        if not self.ndim: raise _errors.BrowserUnsupported('Scalar scatter requires the server')
        if not isinstance(dim, (int, np.integer)) or not -self.ndim <= dim < self.ndim: raise IndexError('scatter dimension out of range')
        dim %= self.ndim
        index = _ops.wrap(index)
        if index.dtype != np.dtype('int64'): raise _errors.BrowserUnsupported('Scatter indices must be int64')
        if index.ndim != self.ndim: raise RuntimeError('scatter index rank must match input')
        if any(index.shape[d] > self.shape[d] for d in range(self.ndim) if d != dim): raise RuntimeError('scatter index shape does not match input')
        if np.any(index._data < 0) or np.any(index._data >= self.shape[dim]): raise IndexError('scatter index out of bounds')
        if isinstance(src, Tensor):
            if src.dtype != self.dtype: raise RuntimeError('scatter source dtype must match input')
            if src.shape != index.shape: raise _errors.BrowserUnsupported('Scatter tensor sources must match index shape')
            if np.any(np.diff(np.sort(index._data, axis=dim), axis=dim) == 0):
                raise _errors.BrowserUnsupported('Scatter with duplicate tensor-source destinations requires the server')
        elif not np.isscalar(src) or np.iscomplexobj(src): raise _errors.BrowserUnsupported('This scatter source requires the server')
        coordinates = list(np.indices(index.shape)); coordinates[dim] = index._data
        coordinates = tuple(coordinates)
        data = self._data.copy(); data[coordinates] = src._data if isinstance(src, Tensor) else src
        overwritten = np.zeros(self.shape, dtype=bool); overwritten[coordinates] = True
        edges = [(self, lambda g: np.where(overwritten, 0, g))]
        if isinstance(src, Tensor): edges.append((src, lambda g: g[coordinates]))
        return _autograd.result(data, *edges, saved=(index,))
    def flip(self, dims):
        if dims is None: raise TypeError('flip requires dimensions')
        axes = () if isinstance(dims, (tuple, list)) and not dims else _ops.reduction_axes(dims, self.ndim)
        return _autograd.result(np.flip(self._data, axis=axes).copy(), (self, lambda g: np.flip(g, axis=axes)))
    def roll(self, shifts, dims=None):
        if dims is None and np.ndim(shifts) and len(shifts) != 1: raise RuntimeError('roll shifts and dimensions must align')
        inverse = -np.asarray(shifts)
        return _autograd.result(np.roll(self._data, shifts, axis=dims), (self, lambda g: np.roll(g, inverse, axis=dims)))
    def triu(self, diagonal=0):
        if self.ndim < 2: raise RuntimeError('triu requires at least two dimensions')
        return _autograd.result(np.triu(self._data, k=diagonal), (self, lambda g: np.triu(g, k=diagonal)))
    def tril(self, diagonal=0):
        if self.ndim < 2: raise RuntimeError('tril requires at least two dimensions')
        return _autograd.result(np.tril(self._data, k=diagonal), (self, lambda g: np.tril(g, k=diagonal)))
    def cumsum(self, dim):
        return _autograd.result(self._data.cumsum(axis=dim), (self, lambda g: np.flip(np.flip(g, axis=dim).cumsum(axis=dim), axis=dim)))
    def cumprod(self, dim, *, dtype=None):
        original_data = self._data
        if not original_data.ndim and dim in (0, -1):
            target = dtype or (np.int64 if original_data.dtype.kind in 'biu' else original_data.dtype)
            return self.to(dtype=target)
        if not -original_data.ndim <= dim < original_data.ndim: raise IndexError('cumprod dimension out of range')
        dim %= original_data.ndim
        if dtype is None and original_data.dtype.kind in 'biu': dtype = np.int64
        value = original_data.cumprod(axis=dim, dtype=dtype)
        def vjp(g):
            current = self._data
            saved = value
            if current is not original_data and np.any(current == 0):
                saved = current.cumprod(axis=dim)
            x, prefix, incoming = (np.moveaxis(array, dim, -1) for array in (current, saved, g))
            grad = np.empty_like(incoming); carry = np.zeros(incoming.shape[:-1], dtype=incoming.dtype)
            for i in reversed(range(x.shape[-1])):
                carry = incoming[..., i] + (carry * x[..., i + 1] if i + 1 < x.shape[-1] else 0)
                grad[..., i] = carry * (prefix[..., i - 1] if i else 1)
            return np.moveaxis(grad, -1, dim)
        return _autograd.result(value, (self, vjp))
    def softmax(self, dim, dtype=None):
        self._real_only()
        x = self.to(dtype=dtype) if dtype is not None else self
        shifted = x._data - x._data.max(axis=dim, keepdims=True)
        exps = np.exp(shifted)
        probabilities = exps / exps.sum(axis=dim, keepdims=True)
        return _autograd.result(probabilities, (x, lambda g: probabilities * (g - (g * probabilities).sum(axis=dim, keepdims=True))))
    def masked_fill(self, mask, value):
        if not isinstance(value, Tensor): value = Tensor(value, dtype=self.dtype)
        return _ops.where(mask, value, self)
    def nan_to_num(self, nan=0., posinf=None, neginf=None):
        return _ops.nan_to_num(self, nan=nan, posinf=posinf, neginf=neginf)
    def clamp(self, min=None, max=None):
        self._real_only()
        if min is None and max is None: raise ValueError('clamp needs at least one bound')
        if isinstance(min, Tensor) or isinstance(max, Tensor): raise _errors.BrowserUnsupported('Tensor clamp bounds require the server')
        mask = np.ones(self.shape, dtype=bool)
        if min is not None: mask &= self._data >= min
        if max is not None: mask &= self._data <= max
        return _autograd.result(np.clip(self._data, min, max), (self, lambda g: g * mask))
    def clamp_min(self, min): return self.clamp(min=min)
    def clamp_max(self, max): return self.clamp(max=max)
    def sin(self):
        source=self.float() if self.dtype.kind in 'biu' else self
        return _autograd.result(np.sin(source._data), (source, lambda g: g * np.cos(source._data)))
    def cos(self):
        source=self.float() if self.dtype.kind in 'biu' else self
        return _autograd.result(np.cos(source._data), (source, lambda g: -g * np.sin(source._data)))
    def acos(self):
        original_data = self._data
        return _autograd.result(np.arccos(original_data), (self, lambda g: -g / np.sqrt(1 - original_data ** 2)))
    def erfinv(self, *, out=None):
        if out is not None or self.dtype not in (np.dtype('float32'), np.dtype('float64')):
            raise _errors.BrowserUnsupported('erfinv requires real float32/float64 without output buffers')
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
        return _autograd.result(value, (self, lambda g: g * (math.sqrt(math.pi)/2) * np.exp(value**2)))
    def tanh(self):
        value = np.tanh(self._data)
        return _autograd.result(value, (self, lambda g: g * (1 - value ** 2)))
    def rsqrt(self): return self ** -.5
    def sigmoid(self):
        self._real_only()
        exp = np.exp(-np.abs(self._data))
        value = np.where(self._data >= 0, 1 / (1 + exp), exp / (1 + exp))
        return _autograd.result(value, (self, lambda g: g * value * (1 - value)))
    def logsigmoid(self):
        original_data = self._data
        if original_data.dtype.kind != 'f': raise RuntimeError('LogSigmoid requires floating inputs')
        if original_data.dtype not in (np.dtype('float32'),np.dtype('float64')):
            raise _errors.BrowserUnsupported('LogSigmoid requires float32/float64 inputs')
        zero = np.zeros((), dtype=original_data.dtype)
        def backward(g):
            if self.dtype != original_data.dtype:
                raise RuntimeError('LogSigmoid backward input and gradient dtypes must match')
            return g * np.exp(-np.logaddexp(zero, original_data))
        return _autograd.result(-np.logaddexp(zero, -original_data), (self, backward))
    def log_softmax(self, dim, dtype=None):
        self._real_only()
        x = self.to(dtype=dtype) if dtype is not None else self
        shifted = x._data - x._data.max(axis=dim, keepdims=True)
        value = shifted - np.log(np.exp(shifted).sum(axis=dim, keepdims=True))
        return _autograd.result(value, (x, lambda g: g - np.exp(value) * g.sum(axis=dim, keepdims=True)))
    def gather(self, dim, index):
        original_data = self._data
        dim %= original_data.ndim
        saved_index = _ops.wrap(index)
        index = saved_index._data
        if index.ndim != original_data.ndim or index.dtype.kind not in 'iu': raise ValueError('gather needs integer indices of the same rank')
        if np.any(index < 0) or np.any(index >= original_data.shape[dim]): raise IndexError('gather index out of bounds')
        if any(index.shape[d] > original_data.shape[d] for d in range(original_data.ndim) if d != dim): raise ValueError('gather dimensions do not match')
        slices = tuple(slice(None) if d == dim else slice(0, index.shape[d]) for d in range(original_data.ndim))
        def vjp(g):
            coordinates = list(np.indices(index.shape)); coordinates[dim] = index
            grad = np.zeros_like(original_data); np.add.at(grad, tuple(coordinates), g)
            return grad
        return _autograd.result(np.take_along_axis(original_data[slices], index, axis=dim), (self, vjp), saved=(saved_index,))
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
        source = self._data
        value = np.std(source, axis=dims, ddof=correction, keepdims=True)
        centered = source - source.mean(axis=dims, keepdims=True)
        def vjp(g):
            if self._data is not source:
                saved = value if keepdim else value.squeeze(axis=dims)
                with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
                    alpha = g / (saved * (count - correction))
                    mask = saved == 0
                    if not keepdim and alpha.ndim:
                        for axis in sorted(dims):alpha,mask = np.expand_dims(alpha,axis),np.expand_dims(mask,axis)
                    current = self._data - self._data.mean(axis=dims,keepdims=True)
                    result = (Tensor(current) * Tensor(alpha))._data
                return np.where(mask,0.,result)
            if not keepdim:
                for axis in sorted(dims): g = np.expand_dims(g, axis)
            derivative = np.divide(centered, value * (count - correction), out=np.zeros_like(centered), where=value != 0)
            return g * derivative
        return _autograd.result(value if keepdim else value.squeeze(axis=dims), (self, vjp))

    def _compare(self, other, op):
        if isinstance(other,(complex,np.complexfloating)): raise _errors.BrowserUnsupported('Complex comparisons require the server')
        # Python scalars use the floating tensor's precision, just like arithmetic.
        # Wrapping first would irreversibly round a float64 boundary to float32.
        if not isinstance(other, (Tensor, np.ndarray)) and self.dtype.kind == 'f':
            other = Tensor(other, dtype=self.dtype)
        other = _ops.wrap(other)
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
    def __and__(self, other): return Tensor(self._data & _ops.wrap(other)._data)
    def __or__(self, other): return Tensor(self._data | _ops.wrap(other)._data)


# Late module references break Tensor/autograd/operator cycles without rebinding functions.
from . import _autograd, _conversion, _device, _errors, _linalg, _ops, _random, _state

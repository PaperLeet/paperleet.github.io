"""NumPy-backed tensor implementation; extracted from PaperLeet."""
import copy
import math
import sys
import weakref
import warnings
from collections import namedtuple
from statistics import NormalDist
import numpy as np
from . import _layout, _statistics, _reductions, _matmul

_WASM_MATMUL = sys.platform == 'emscripten'


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


def storage_owner(data):
    """Resolve ndarray bases and NumPy's metadata-only stride wrappers."""
    owner = data
    while True:
        base = owner.base
        if isinstance(base, np.ndarray): owner = base
        elif isinstance(getattr(base, 'base', None), np.ndarray): owner = base.base
        else: return owner


def storage_version(data):
    """Views, detach and dtype-preserving casts share their NumPy allocation's counter."""
    owner = data
    while isinstance(owner.base, np.ndarray): owner = owner.base
    if owner.base is not None: owner = storage_owner(owner)
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
    _retain_grad = False
    _view_base = None
    _no_grad_view = False
    _backward_hooks = None
    _post_accumulate_grad_hooks = None

    def __init__(self, data, dtype=None, requires_grad=False, *, _metadata_owned=False):
        self._data = np.asarray(data, dtype=dtype)
        if self._data.dtype.char not in '?bBhHiIlLqQefdFD':
            raise TypeError(f'Unsupported Tensor dtype: {self._data.dtype} ({self._data.dtype.type.__name__})')
        self._storage = storage_version(self._data)
        # Sharing bytes must not share a mutable ndarray descriptor. Internal
        # results/copies already own their descriptors; they still validate dtype.
        if not _metadata_owned:
            self._data = self._data.view()
        self._grad_node = None
        self._requires_grad = bool(requires_grad)
        if requires_grad and self.dtype.kind == 'c': raise _errors.BrowserUnsupported('Complex autograd requires the server')
        if requires_grad and self.dtype.kind != 'f':
            raise RuntimeError('Only floating point tensors can require gradients')
        self.grad = None

    @property
    def requires_grad(self):
        flag=self._requires_grad
        return self._view_base.requires_grad if flag is None else flag

    @requires_grad.setter
    def requires_grad(self, flag):
        if type(flag) is not bool:raise RuntimeError('requires_grad must be a bool')
        if self._grad_node is not None:raise RuntimeError('Only leaf tensors can change requires_grad flags')
        self.requires_grad_(flag)

    @property
    def _parents(self):
        node = self._grad_node
        return () if node is None else node._parents

    @property
    def _joint_vjp(self):
        node = self._grad_node
        return None if node is None else node.joint

    retains_grad = property(lambda self: bool(self._retain_grad and self._grad_node is not None))

    def retain_grad(self):
        if _state._jvp_active:raise _errors.BrowserUnsupported('retain_grad inside a forward AD transform is unsupported')
        if not self.requires_grad:raise RuntimeError('Cannot retain_grad on a tensor that does not require grad')
        if self._grad_node is not None:
            self._retain_grad = True
            _tensor_hooks.callbacks_for(self)

    def register_hook(self, hook):
        return _tensor_hooks.register(self, hook)

    def register_post_accumulate_grad_hook(self, hook):
        return _tensor_hooks.register(self, hook, post=True)

    shape = property(lambda self: self._data.shape)
    ndim = property(lambda self: self._data.ndim)
    dtype = property(lambda self: self._data.dtype)
    device = property(lambda self: _device._CPU)
    is_leaf = property(lambda self: self._grad_node is None)
    _version = property(lambda self: self._storage.version)
    @property
    def T(self):
        if self.ndim==0:
            if not getattr(_state,'_scalar_t_warning_emitted',False):
                warnings.warn('Tensor.T is deprecated on 0-D tensors.',UserWarning,stacklevel=2)
                _state._scalar_t_warning_emitted=True
        elif self.ndim>2 and not getattr(_state,'_high_rank_t_warning_emitted',False):
            warnings.warn('Tensor.T on tensors with more than two dimensions is deprecated.',UserWarning,stacklevel=2)
            _state._high_rank_t_warning_emitted=True
        return self.permute(tuple(reversed(range(self.ndim))))
    @property
    def mT(self):
        if self.ndim==0:
            if not getattr(_state,'_scalar_mt_warning_emitted',False):
                warnings.warn('Tensor.mT is deprecated on 0-D tensors.',UserWarning,stacklevel=2)
                _state._scalar_mt_warning_emitted=True
            return self
        if self.ndim==1:raise RuntimeError('Tensor.mT requires a matrix or a batch of matrices')
        return self.transpose(-2,-1)
    data = property(lambda self: self.detach())

    def __getattr__(self, name):
        return _errors.missing_attribute(self, name, f'Tensor.{name}')
    def _real_only(self):
        if self.dtype.kind == 'c': raise _errors.BrowserUnsupported('This complex operation requires the server')
    def __array__(self, dtype=None, copy=None):
        if _state._functional_ad_active and _state._grad_enabled and self.requires_grad:
            raise RuntimeError('A functional AD transform cannot export a recorded Tensor to NumPy; detach explicitly')
        exported = self._data.view()
        return np.array(exported, dtype=dtype, copy=copy) if copy is not None else np.asarray(exported, dtype=dtype)
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
    def numpy(self, *, force=False):
        if force is False:
            if _state._functional_ad_active and _state._grad_enabled and self.requires_grad:
                raise RuntimeError('A functional AD transform cannot export a recorded Tensor to NumPy; detach explicitly')
        elif force is not True:
            raise TypeError('numpy force must be bool')
        return self._data.view()
    def numel(self): return self._data.size
    def data_ptr(self): return int(self._data.__array_interface__['data'][0]) if self.numel() else 0
    def dim(self): return self.ndim
    def stride(self, dim=None): return _layout.stride(self._data, dim)
    def element_size(self): return self._data.itemsize
    def is_floating_point(self): return self.dtype.kind == 'f'
    def count_nonzero(self, dim=None): return _ops.count_nonzero(self, dim)
    def size(self, dim=None): return self.shape if dim is None else self.shape[dim]
    def detach(self): return Tensor(self._data)
    def clone(self, *, memory_format=None):
        fmt = _conversion.memory_format(memory_format)
        data = self._data.copy(order='C' if fmt is _conversion.MemoryFormat.contiguous_format else 'K')
        return _autograd.result(data, (self, lambda g: g), saves_tensors=False)
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
    def where(self, condition, other): return _ops.where(condition, self, other)
    def new_zeros(self, *shape, **kw): return _random.create('zeros', *shape, **_ops.inherited_options(self.dtype,kw))
    def new_ones(self, *shape, **kw): return _random.create('ones', *shape, **_ops.inherited_options(self.dtype,kw))
    def new_empty(self, *shape, **kw): return _random.create('empty', *shape, **_ops.inherited_options(self.dtype,kw))
    def new_full(self, size, fill_value, **kw): return _ops.full(size, fill_value, **_ops.inherited_options(self.dtype,kw))
    def new_tensor(self, data, *, dtype=None, device=None, requires_grad=False):
        return _ops.tensor(data, dtype=self.dtype if dtype is None else dtype,
                           device=device, requires_grad=requires_grad)
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
        return _autograd.result(data, (self, lambda g: g), saves_tensors=False) if data.dtype.kind in 'fc' else Tensor(data)

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
        owner = storage_owner(self._data)
        previous = _state._storages.get(id(owner))
        if previous is not None and previous[0]() is owner:
            _state._storages[id(owner)] = (previous[0], self._storage)

    def requires_grad_(self, requires_grad=True):
        if type(requires_grad) is not bool:raise TypeError('requires_grad_ requires a bool')
        if _state._jvp_active: raise _errors.BrowserUnsupported('Changing gradient flags inside JVP requires the server')
        if requires_grad and self.dtype.kind == 'c': raise _errors.BrowserUnsupported('Complex autograd requires the server')
        if requires_grad and self.dtype.kind != 'f': raise RuntimeError('Only floating tensors can require grad')
        if self._grad_node is not None and not requires_grad:raise RuntimeError('Only leaf tensors can change requires_grad flags')
        if not requires_grad and self._grad_node is None and self._view_base is not None:
            self._requires_grad = None
            self._no_grad_view = True
            return self
        if requires_grad:self._no_grad_view = False
        self._requires_grad = requires_grad
        return self

    def backward(self, gradient=None, *, retain_graph=None, create_graph=False, inputs=None):
        return _autograd.autograd_backward(self, gradient, retain_graph, create_graph, inputs=inputs)

    def _accumulate_gradient(self, grad):
        if self.grad is None:
            self.grad = Tensor(_layout.gradient_copy(grad, self._data if self._grad_node is None else None),
                               _metadata_owned=True)
        else:
            if not isinstance(self.grad, Tensor) or self.grad.shape != grad.shape or self.grad.dtype.kind != 'f':
                raise RuntimeError('Existing gradient must be a floating Tensor of the correct shape')
            with np.errstate(all='ignore'):
                if self._grad_node is None:
                    self.grad._data[...] += grad
                    self.grad._storage.bump()
                else:
                    self.grad = Tensor(_layout.contiguous_empty(self.grad._data + grad),
                                       _metadata_owned=True)

    def __getitem__(self, index):
        original_data = self._data
        original_shape,original_dtype = self.shape,self.dtype
        saved = []
        def convert(key):
            if isinstance(key, Tensor):
                if key.ndim==0 and key.dtype in (np.dtype('int32'),np.dtype('int64')):
                    return int(key._data.item())
                saved.append(key);return key._data
            if isinstance(key, tuple): return tuple(convert(k) for k in key)
            return key
        index = convert(index)
        keys = index if isinstance(index, tuple) else (index,)
        basic = all(key is None or key is Ellipsis or isinstance(key, slice) or
                    isinstance(key, (int, np.integer)) and not isinstance(key, (bool, np.bool_))
                    for key in keys)
        def vjp(g):
            grad = np.zeros(original_shape,dtype=original_dtype)
            # Basic slices address each destination once. Advanced indexing can
            # repeat destinations, and must retain unbuffered scatter addition.
            if basic:
                grad[index] += g
            else:
                np.add.at(grad, index, g)
            return grad
        data=original_data[index]
        if basic and not isinstance(data,np.ndarray):
            # NumPy returns a scalar object for an all-integer selection.
            # Preserve Torch's zero-dimensional view and shared version counter.
            slices=[];axis=0
            for key in keys:
                if key is Ellipsis:slices.append(key)
                else:
                    position=int(key);position=position if position>=0 else position+original_shape[axis]
                    slices.append(slice(position,position+1));axis+=1
            data=original_data[tuple(slices)].reshape(()) if slices else original_data.view().reshape(())
        if basic:return _autograd.view_result(data,self,vjp)
        return _autograd.result(data,(self,vjp),saved=saved)

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

    def copy_(self, src, non_blocking=False):
        if not isinstance(src, Tensor):raise TypeError('copy_ source must be a Tensor')
        if type(non_blocking) is not bool:raise TypeError('non_blocking must be bool')
        if _state._jvp_active:raise _errors.BrowserUnsupported('Tensor copy_ inside forward AD is unsupported')
        if self.dtype.kind not in 'biufc' or src.dtype.kind not in 'biufc':
            raise _errors.BrowserUnsupported('copy_ requires boolean or numeric Tensor storage')
        if _state._grad_enabled and (self.requires_grad or src.requires_grad):
            raise _errors.BrowserUnsupported('Differentiable in-place copy_ requires the server')
        destination,source = self._data,src._data
        try:np.broadcast_to(source,destination.shape)
        except ValueError as error:raise RuntimeError('copy_ source cannot broadcast to destination shape') from error
        same = (destination.dtype==source.dtype and destination.shape==source.shape and
                destination.strides==source.strides and
                destination.__array_interface__['data'][0]==source.__array_interface__['data'][0])
        if not same and destination.size:
            if any(size>1 and stride==0 for size,stride in zip(destination.shape,destination.strides)):
                raise RuntimeError('copy_ destination has overlapping elements')
            try:overlap=np.shares_memory(destination,source,max_work=1000)
            except np.exceptions.TooHardError as error:
                raise _errors.BrowserUnsupported('copy_ storage overlap cannot be determined within its budget') from error
            if overlap:raise RuntimeError('copy_ source and destination overlap; clone the source first')
            writable=destination.view()
            try:writable.setflags(write=True)
            except ValueError as error:raise _errors.BrowserUnsupported('copy_ destination has read-only backing storage') from error
            if source.dtype.kind=='c' and destination.dtype.kind not in 'cb':
                if not _state._copy_complex_warning_emitted:
                    import warnings
                    _state._copy_complex_warning_emitted=True
                    warnings.warn('Casting complex values to real discards the imaginary part',UserWarning,stacklevel=2)
                source=source.real
            if source.dtype==destination.dtype:
                np.copyto(writable,source,casting='unsafe')
            else:
                with np.errstate(all='ignore'):
                    np.copyto(writable,source,casting='unsafe')
        self._storage.bump()
        return self

    def __deepcopy__(self, memo):
        if self._parents:
            raise _errors.BrowserUnsupported('Copying non-leaf autograd tensors requires the server')
        owner = storage_owner(self._data)
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

    def _binary(self, other, op, left, right, *, saves_tensors=True, floating=False, subtract=False, reverse=False):
        other, a, b, dtype = _arithmetic.operands(self, other, floating=floating, subtract=subtract)
        if dtype.kind == 'u' and dtype.itemsize > 1 and (op in (np.add, np.subtract) or subtract):
            raise _errors.BrowserUnsupported('This unsigned arithmetic kernel requires the server')
        with np.errstate(all='ignore'):
            try:
                first, second = (b, a) if reverse else (a, b)
                if dtype.kind == 'c' and op in (np.add, np.subtract):
                    sign = np.asarray(1 if op is np.add else -1, dtype=dtype)
                    data = np.add(first, np.multiply(second, sign))
                else: data = op(first, second)
            except ValueError as error: raise RuntimeError('Arithmetic operands have incompatible shapes') from error
        if (not _state._grad_enabled or not self.requires_grad and (other is None or not other.requires_grad)) and not isinstance(self, _autograd.ForwardTensor) and not isinstance(other, _autograd.ForwardTensor):
            return _autograd.result(data, saves_tensors=False)
        if not saves_tensors:
            left_shape, right_shape = a.shape, b.shape
            left_edge = (self, lambda g: _ops.unbroadcast(left(g,None,None),left_shape))
            if other is None: return _autograd.result(data, left_edge, saves_tensors=False)
            return _autograd.result(data, left_edge,
                (other, lambda g: _ops.unbroadcast(right(g,None,None),right_shape)), saves_tensors=False)
        left_edge = (self, lambda g: _ops.unbroadcast(left(g,a,b),a.shape))
        if other is None: return _autograd.result(data, left_edge)
        return _autograd.result(data, left_edge,
            (other, lambda g: _ops.unbroadcast(right(g,a,b),b.shape)))
    def __add__(self, other): return self._binary(other, np.add, lambda g,a,b:g, lambda g,a,b:g, saves_tensors=False)
    __radd__ = __add__
    def __sub__(self, other): return self._binary(other, np.subtract, lambda g,a,b:g, lambda g,a,b:-g, saves_tensors=False, subtract=True)
    def __rsub__(self, other): return self._binary(other, np.subtract, lambda g,a,b:-g, lambda g,a,b:g, saves_tensors=False, subtract=True, reverse=True)
    def __mul__(self, other): return self._binary(other, np.multiply, lambda g,a,b:g*b, lambda g,a,b:g*a)
    __rmul__ = __mul__
    def __truediv__(self, other): return self._binary(other, np.divide, lambda g,a,b:g/b, lambda g,a,b:-g*((a/b)/b), floating=True)
    def __rtruediv__(self, other):
        if type(self) is Tensor and type(other) in (bool, int, float, complex):
            return _arithmetic.reflected_number_divide(self, other)
        return self.reciprocal() * other
    def reciprocal(self):
        source = self._data
        with np.errstate(all='ignore'):
            data = _arithmetic.reciprocal_values(source)
        value = _autograd.result(data, (self, lambda g: g * -(data * data)), versioned=())
        if value.requires_grad:
            # Reciprocal backward saves its output, not the input values.
            value._grad_node.versions = ((value._storage, value._storage.version),)
        return value
    def add(self, other, *, alpha=1): return _arithmetic.add(self, other, alpha=alpha)
    def sub(self, other, *, alpha=1): return _arithmetic.sub(self, other, alpha=alpha)
    def mul(self, other): return _arithmetic.mul(self, other)
    def div(self, other, *, rounding_mode=None): return _arithmetic.div(self, other, rounding_mode=rounding_mode)
    def true_divide(self, other): return _arithmetic.true_divide(self, other)
    subtract, multiply, divide = sub, mul, div
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
        return self._binary(other,np.remainder,lambda g,a,b:g,lambda g,a,b:-g*np.floor(a/b),
                            saves_tensors=_ops.wrap(other).requires_grad)
    __mod__ = remainder
    def __rmod__(self, other):
        self._real_only(); _ops.wrap(other)._real_only()
        if self.dtype.kind=='b' and _ops.wrap(other).dtype.kind=='b': raise _errors.BrowserUnsupported('Boolean remainder requires the server')
        if self.dtype.kind in 'biu' and _ops.wrap(other).dtype.kind in 'biu' and np.any(self._data==0):
            raise RuntimeError('Integer division by zero')
        return self._binary(other,lambda a,b:np.remainder(b,a),lambda g,a,b:-g*np.floor(b/a),lambda g,a,b:g)
    def __neg__(self): return _autograd.result(-self._data, (self, lambda g: -g), saves_tensors=False)
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
        if self.dtype.kind in 'biu' and isinstance(exponent,(float,np.floating)):
            return self.to(dtype=_defaults._float_dtype) ** exponent
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
        if self.dtype.kind in 'biu':return self.to(dtype=_defaults._float_dtype).exp()
        data = np.exp(self._data)
        return _autograd.result(data, (self, lambda g: g * data))
    def log(self):
        if self.dtype.kind in 'biu':return self.to(dtype=_defaults._float_dtype).log()
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

    def mm(self, mat2):
        return _ops.mm(self, mat2)
    def bmm(self, mat2):
        return _ops.bmm(self, mat2)
    def matmul(self, other):
        return _ops.matmul(self, other)
    def dot(self, tensor):
        return _ops.dot(self, tensor)
    def mv(self, vec):
        return _ops.mv(self, vec)
    def __matmul__(self, other):
        if not isinstance(self, Tensor):raise TypeError('matmul receiver must be a Tensor')
        if not isinstance(other, Tensor):return NotImplemented
        a, b = self._data, other._data
        dtype = a.dtype
        needs_grad = _state._grad_enabled and (self.requires_grad or other.requires_grad)
        unsupported_dtype = dtype.kind in 'bu' and dtype != _ops._BYTE_DTYPE
        if (dtype != b.dtype or unsupported_dtype
                or needs_grad and (a.ndim > 2 or b.ndim > 2) and (not a.size or not b.size)):
            special = _ops._matmul_special(self, other)
            if special is not None:return _ops._empty_product(self, other, *special)
        try:value = _matmul.product(a, b) if _WASM_MATMUL and a.size >= 4096 else a @ b
        except ValueError as error:raise RuntimeError(str(error)) from None
        if not value.size:value = _layout.contiguous_empty(value)
        # The ordinary inference route needs no pullback closures or saved
        # factors. Subclasses keep result()'s forward-transform safeguards.
        if not needs_grad and type(self) is Tensor and type(other) is Tensor:
            return Tensor(value, _metadata_owned=True)
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
            if _WASM_MATMUL and a.size >= 4096:
                grad = _ops.unbroadcast(_matmul.product(promote(g), np.swapaxes(current, -1, -2)), aa.shape)
            else:
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
            if _WASM_MATMUL and a.size >= 4096:
                grad = _ops.unbroadcast(_matmul.product(np.swapaxes(current, -1, -2), promote(g)), bb.shape)
            else:
                grad = _ops.unbroadcast(np.swapaxes(current, -1, -2) @ promote(g), bb.shape)
            return grad.reshape(b.shape)
        output=_autograd.result(value, (self, left), (other, right), versioned=())
        if output.requires_grad:
            # Each derivative saves only its opposite operand. The result is
            # never read by either pullback, so its mutation is harmless.
            counters=set()
            if self.requires_grad:counters.add(other._storage)
            if other.requires_grad:counters.add(self._storage)
            output._grad_node.versions=tuple((counter,counter.version) for counter in counters)
        return output
    def __rmatmul__(self, other):
        if not isinstance(self, Tensor):raise TypeError('matmul receiver must be a Tensor')
        return other @ self if isinstance(other, Tensor) else NotImplemented

    def __array_wrap__(self, array, context=None, return_scalar=False):
        # NumPy matmul inference follows Python's NotImplemented fallback.
        # __array__ keeps the existing guard against implicit graph escapes.
        if context is not None and context[0] is np.matmul:
            if _state._grad_enabled and self.requires_grad:
                raise RuntimeError('NumPy matmul requires detached tensors')
            return Tensor(array, _metadata_owned=True)
        return array[()] if return_scalar else array

    def reshape(self, *shape):
        original_data = self._data
        original_shape = self.shape
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)): shape = tuple(shape[0])
        return _autograd.view_result(original_data.reshape(shape),self,lambda g:g.reshape(original_shape))
    def view(self, *shape, **kwargs):
        original_data = self._data
        original_shape = original_data.shape
        data = None
        if not kwargs and original_data.size > 1:
            target = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
            if len(target) == 2 and type(target[0]) is int and type(target[1]) is int:
                a, b = target
                if a == -1 and b > 0 and original_data.size % b == 0:
                    target = (original_data.size // b, b)
                elif b == -1 and a > 0 and original_data.size % a == 0:
                    target = (a, original_data.size // a)
                if target[0] > 0 and target[1] > 0 and target[0] * target[1] == original_data.size:
                    # Geometry proves these reshapes cannot allocate a data copy.
                    # Keep the common two-integer call free of generic parsers.
                    if original_data.ndim == 1 and original_data.strides[0] % original_data.itemsize == 0:
                        data = original_data.reshape(target)
                    elif original_data.flags.c_contiguous and 1 not in original_shape:
                        data = original_data.reshape(target)
                    elif tuple(target) == original_shape and 1 not in original_shape:
                        strides = original_data.strides
                        if strides[0] % original_data.itemsize == 0 and strides[1] % original_data.itemsize == 0:
                            data = original_data.view()
        if data is None:
            args = _layout.arguments(shape, kwargs)
            data = _layout.view(original_data, *args)
        return _autograd.view_result(data,self,lambda g:g.reshape(original_shape))
    def view_as(self, other):
        if not isinstance(other, Tensor): raise TypeError('view_as requires a Tensor')
        return self.view(other.shape)
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
        start=_layout.dimension(start_dim,self.ndim);end=_layout.dimension(end_dim,self.ndim)
        if start>end:raise RuntimeError('flatten start_dim cannot follow end_dim')
        if not self.ndim:return self.reshape(1)
        if start==end:return self
        shape=(*self.shape[:start],math.prod(self.shape[start:end+1]),*self.shape[end+1:])
        return self.reshape(shape)
    def unsqueeze(self, dim):
        original_data = self._data
        original_shape = self.shape
        axis=_layout.dimension(dim,self.ndim,insertion=True)
        shape=(*original_shape[:axis],1,*original_shape[axis:])
        step=original_data.itemsize if axis==self.ndim else original_data.strides[axis]*original_shape[axis]
        strides=(*original_data.strides[:axis],step,*original_data.strides[axis:])
        value=_layout.metadata_view(original_data,shape,strides,original_data.reshape(shape))
        return _autograd.view_result(value,self,lambda g:g.reshape(original_shape))
    def squeeze(self, dim=_layout.UNSPECIFIED_DIM):
        original_data = self._data
        original_shape = self.shape
        if dim is _layout.UNSPECIFIED_DIM:axes=None
        elif type(dim) is int:
            axis=_layout.dimension(dim,self.ndim)
            axes=(axis,) if self.ndim and original_shape[axis]==1 else ()
        else:
            dims=dim if isinstance(dim,(tuple,list)) else (dim,)
            dims=tuple(_layout.dimension(v,self.ndim) for v in dims)
            if len(set(dims))!=len(dims):raise RuntimeError('Squeeze dimensions must be unique')
            axes=tuple(v for v in dims if self.ndim and original_shape[v]==1)
        return _autograd.view_result(np.squeeze(original_data,axis=axes),self,lambda g:g.reshape(original_shape))
    def permute(self, *dims, **kwargs):
        dims=_layout.permutation(dims,kwargs,self.ndim)
        return _autograd.view_result(self._data.transpose(dims),self,lambda g:g.transpose(_layout.inverse_permutation(dims)))
    def transpose(self, dim0, dim1):
        ndim=self.ndim;extent=ndim or 1
        if type(dim0) is int and type(dim1) is int and -extent<=dim0<extent and -extent<=dim1<extent:
            dim0%=extent;dim1%=extent
        else:dim0,dim1=_layout.transpose_dimensions(dim0,dim1,ndim)
        if not ndim:
            return _autograd.view_result(self._data.view(),self,lambda g:g)
        return _autograd.view_result(self._data.swapaxes(dim0,dim1),self,lambda g:g.swapaxes(dim0,dim1))
    def t(self):
        if self.ndim>2:raise RuntimeError('t() expects a tensor with at most two dimensions')
        return self.transpose(0,1 if self.ndim==2 else 0)
    def swapdims(self, dim0, dim1):return self.transpose(dim0,dim1)
    def swapaxes(self, axis0, axis1):return self.transpose(axis0,axis1)
    def movedim(self, source, destination):
        return self.permute(_layout.movement(source,destination,self.ndim))
    def moveaxis(self, source, destination):return self.movedim(source,destination)
    def expand(self, *shape, **kwargs):
        original_data = self._data
        original_shape = self.shape
        target,strides=_layout.expansion(shape,kwargs,original_shape,original_data.strides)
        value=_layout.metadata_view(original_data,target,strides)
        if value.flags.writeable:
            # Keep expansion read-only while retaining a writable ndarray base
            # for copy_'s separately checked, nonoverlapping destinations.
            value=value.view()
            value.flags.writeable=False
        return _autograd.view_result(value,self,lambda g:_ops.unbroadcast(g,original_shape))
    def expand_as(self, other): return self.expand(other.shape)
    def repeat(self, *repeats):
        original_data = self._data
        original_shape = self.shape
        if len(repeats) == 1 and isinstance(repeats[0], (tuple, list)): repeats = tuple(repeats[0])
        shape = (1,) * (len(repeats) - original_data.ndim) + original_data.shape
        interleaved = tuple(v for pair in zip(repeats, shape) for v in pair)
        return _autograd.result(np.tile(original_data, repeats), (self, lambda g: g.reshape(interleaved).sum(axis=tuple(range(0, len(interleaved), 2))).reshape(original_shape)), saves_tensors=False)
    def repeat_interleave(self, repeats, dim=None, *, output_size=None):
        source = self.flatten() if dim is None else self
        dim = 0 if dim is None else dim
        if not -source.ndim <= dim < source.ndim: raise IndexError('repeat_interleave dimension out of range')
        dim %= source.ndim
        if isinstance(repeats,(int,np.integer)) and not isinstance(repeats,(bool,np.bool_)):
            count=int(repeats)
            if count<0:raise RuntimeError('Invalid repetition counts')
            if output_size is not None and output_size!=source.shape[dim]*count:
                raise RuntimeError('Incorrect repeat_interleave output_size')
            shape=list(source.shape);shape.insert(dim+1,count)
            return _autograd.result(np.repeat(source._data,count,axis=dim),
                (source,lambda g:g.reshape(shape).sum(axis=dim+1)),saves_tensors=False)
        counts = _ops.wrap(repeats)._data
        if counts.dtype.kind not in 'iu' or counts.ndim > 1 or np.any(counts < 0): raise RuntimeError('Invalid repetition counts')
        if np.any(counts > np.iinfo(np.intp).max):
            raise _errors.BrowserUnsupported('Repetition counts exceed this runtime index range')
        # wasm32 uses int32 indices even though torch.long remains int64.
        indices = np.repeat(np.arange(source.shape[dim], dtype=np.intp), counts.astype(np.intp))
        if output_size is not None and output_size != len(indices): raise RuntimeError('Incorrect repeat_interleave output_size')
        index = [slice(None)] * source.ndim; index[dim] = indices
        return source[tuple(index)]

    def sum(self, dim=_layout.UNSPECIFIED_DIM, keepdim=_layout.UNSPECIFIED_DIM, *, dtype=None):
        requested, keepdim = _reductions.arguments(dim,keepdim,name='sum')
        if dtype is not None and np.dtype(dtype).kind not in 'fc' and self.requires_grad and _state._grad_enabled:
            raise RuntimeError('Autograd does not support a non-floating sum dtype')
        original_data = self._data
        original_shape = self.shape
        dims = _reductions.axes(requested,self.ndim)
        def vjp(g):
            if not keepdim:
                for axis in sorted(dims): g = np.expand_dims(g, axis)
            return np.broadcast_to(g, original_shape)
        if dtype is None and original_data.dtype.kind in 'biu': dtype = np.int64
        return _autograd.result(original_data.sum(axis=dims, keepdims=keepdim, dtype=dtype), (self, vjp), saves_tensors=False)
    def mean(self, dim=_layout.UNSPECIFIED_DIM, keepdim=_layout.UNSPECIFIED_DIM, *, dtype=None):
        requested, keepdim = _reductions.arguments(dim,keepdim,name='mean')
        target = self.dtype if dtype is None else np.dtype(dtype)
        if target.kind not in 'fc':raise RuntimeError('mean requires floating or complex dtype')
        dims = _reductions.axes(requested,self.ndim)
        shape = self.shape
        count = math.prod(shape[d] for d in dims)
        def vjp(g):
            if not keepdim:
                for axis in sorted(dims):g=np.expand_dims(g,axis)
            if g.dtype == np.dtype('float16'):
                # Divide in opmath precision, then round before the edge casts
                # back into the input dtype. The count may exceed half's range.
                gradient = np.divide(g,count,dtype=np.float32).astype(g.dtype)
            else:gradient = np.divide(g,count,dtype=g.dtype)
            return np.broadcast_to(gradient,shape)
        with np.errstate(all='ignore'):
            half = target == np.dtype('float16')
            accumulator = np.float32 if half else target
            # Native CPU half-output means convert input directly to float32;
            # they do not round wider input to half before the accumulation.
            if half:value = self._data.sum(axis=dims,keepdims=keepdim,dtype=accumulator)/np.asarray(count,dtype=accumulator)
            elif count:value = self._data.mean(axis=dims,keepdims=keepdim,dtype=accumulator)
            else:value = self._data.sum(axis=dims,keepdims=keepdim,dtype=accumulator)/np.asarray(0,dtype=accumulator)
            if half:value = value.astype(target,copy=False)
        return _autograd.result(value,(self,vjp),saves_tensors=False)
    def nansum(self, dim=None, keepdim=False, *, dtype=None):
        return _reductions.nan_reduce(self,dim,keepdim,dtype=dtype)
    def nanmean(self, dim=None, keepdim=False, *, dtype=None):
        return _reductions.nan_reduce(self,dim,keepdim,dtype=dtype,mean=True)
    def trace(self):
        original_data = self._data
        if original_data.ndim != 2: raise RuntimeError('trace expects a matrix')
        self._real_only()
        shape,dtype = self.shape,self.dtype
        return _autograd.result(np.trace(original_data), (self, lambda g: np.eye(*shape, dtype=dtype) * g), saves_tensors=False)
    def det(self): return _linalg.det(self)
    def inverse(self): return _linalg.inv(self)
    def slogdet(self): return _linalg.slogdet(self)
    def logdet(self): return _linalg.logdet(self)
    def cholesky_solve(self, input2, upper=False):
        return _cholesky.cholesky_solve(self, input2, upper)

    def cholesky_inverse(self, upper=False):
        return _cholesky.cholesky_inverse(self, upper)
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
        shape,dtype = self.shape,self.dtype
        return _autograd.result(np.rint(original_data),(self,lambda g: np.zeros(shape,dtype=dtype)),saves_tensors=False)
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
        if self.dtype.kind in 'biu':return self.to(dtype=_defaults._float_dtype).logsumexp(dim,keepdim)
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
        shape,dtype = self.shape,self.dtype
        if original_data.ndim < 2: raise RuntimeError('diagonal requires at least two dimensions')
        if not isinstance(offset, (int, np.integer)): raise TypeError('Diagonal offset must be an integer')
        axes = _ops.reduction_axes((dim1, dim2), original_data.ndim)
        value = np.diagonal(original_data, offset=int(offset), axis1=axes[0], axis2=axes[1])
        if original_data.flags.writeable: value.setflags(write=True)
        length = value.shape[-1]
        def vjp(g):
            gradient = np.zeros(shape,dtype=dtype)
            moved = np.moveaxis(gradient, axes, (-2, -1))
            index = np.arange(length)
            moved[..., index + max(-int(offset), 0), index + max(int(offset), 0)] = g
            return gradient
        return _autograd.result(value, (self, vjp),saves_tensors=False)
    def diag(self, diagonal=0, *, out=None):
        original_data = self._data
        shape,dtype,rank = self.shape,self.dtype,self.ndim
        if out is not None: raise _errors.BrowserUnsupported('diag out arguments require the server')
        if original_data.ndim not in (1, 2): raise RuntimeError('diag expects a vector or matrix')
        value = np.diag(original_data, k=diagonal)
        length = value.size
        def vjp(g):
            if rank == 1: return np.diag(g, k=diagonal)
            gradient = np.zeros(shape,dtype=dtype)
            positions = np.arange(length, dtype=np.intp)
            gradient[positions + max(-diagonal, 0), positions + max(diagonal, 0)] = g
            return gradient
        return _autograd.result(value, (self, vjp),saves_tensors=False)
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
    def argsort(self, *args, dim=_layout.UNSPECIFIED_DIM, descending=_layout.UNSPECIFIED_DIM,
                stable=_layout.UNSPECIFIED_DIM):
        dim,descending,stable=_indexing.sort_arguments(args,dim,descending,stable)
        return _indexing.argsort(self,dim,descending,stable=stable)
    def sort(self, *args, dim=_layout.UNSPECIFIED_DIM, descending=_layout.UNSPECIFIED_DIM,
             stable=_layout.UNSPECIFIED_DIM, out=None):
        dim,descending,stable=_indexing.sort_arguments(args,dim,descending,stable)
        return _indexing.sort(self,dim,descending,stable=stable,out=out)
    def topk(self, k, dim=-1, largest=True, sorted=True, *, out=None):
        return _indexing.topk(self,k,dim,largest,sorted,out=out)
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
        return _autograd.result(np.flip(self._data, axis=axes).copy(), (self, lambda g: np.flip(g, axis=axes)),saves_tensors=False)
    def roll(self, shifts, dims=None):
        if dims is None and np.ndim(shifts) and len(shifts) != 1: raise RuntimeError('roll shifts and dimensions must align')
        inverse = -np.asarray(shifts)
        return _autograd.result(np.roll(self._data, shifts, axis=dims), (self, lambda g: np.roll(g, inverse, axis=dims)),saves_tensors=False)
    def triu(self, diagonal=0):
        if self.ndim < 2: raise RuntimeError('triu requires at least two dimensions')
        return _autograd.result(np.triu(self._data, k=diagonal), (self, lambda g: np.triu(g, k=diagonal)),saves_tensors=False)
    def tril(self, diagonal=0):
        if self.ndim < 2: raise RuntimeError('tril requires at least two dimensions')
        return _autograd.result(np.tril(self._data, k=diagonal), (self, lambda g: np.tril(g, k=diagonal)),saves_tensors=False)
    def cumsum(self, dim):
        return _autograd.result(self._data.cumsum(axis=dim), (self, lambda g: np.flip(np.flip(g, axis=dim).cumsum(axis=dim), axis=dim)),saves_tensors=False)
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
        source=self.to(dtype=_defaults._float_dtype) if self.dtype.kind in 'biu' else self
        return _autograd.result(np.sin(source._data), (source, lambda g: g * np.cos(source._data)))
    def cos(self):
        source=self.to(dtype=_defaults._float_dtype) if self.dtype.kind in 'biu' else self
        return _autograd.result(np.cos(source._data), (source, lambda g: -g * np.sin(source._data)))
    def acos(self):
        if self.dtype.kind in 'biu':return self.to(dtype=_defaults._float_dtype).acos()
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
        if self.dtype.kind in 'biu':return self.to(dtype=_defaults._float_dtype).tanh()
        value = np.tanh(self._data)
        return _autograd.result(value, (self, lambda g: g * (1 - value ** 2)))
    def rsqrt(self): return self ** -.5
    def sigmoid(self):
        if self.dtype.kind in 'biu':return self.to(dtype=_defaults._float_dtype).sigmoid()
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
    def gather(self, dim, index, *, sparse_grad=False):
        return _indexing.gather(self,dim,index,sparse_grad=sparse_grad)
    def var(self, *args, dim=_layout.UNSPECIFIED_DIM, unbiased=_layout.UNSPECIFIED_DIM,
            keepdim=_layout.UNSPECIFIED_DIM, correction=_layout.UNSPECIFIED_DIM):
        return _statistics.reduce(self, *args, dim=dim, unbiased=unbiased,
                                  keepdim=keepdim, correction=correction)
    def std(self, *args, dim=_layout.UNSPECIFIED_DIM, unbiased=_layout.UNSPECIFIED_DIM,
            keepdim=_layout.UNSPECIFIED_DIM, correction=_layout.UNSPECIFIED_DIM):
        return _statistics.reduce(self, *args, dim=dim, unbiased=unbiased,
                                  keepdim=keepdim, correction=correction, kind='std')

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
from . import _arithmetic, _autograd, _cholesky, _conversion, _defaults, _device, _errors, _linalg, _ops, _random, _state, _tensor_hooks, _indexing

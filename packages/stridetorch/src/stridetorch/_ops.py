"""NumPy-backed ops implementation; extracted from PaperLeet."""
import numpy as np
import warnings
import math
from functools import reduce
from operator import or_
import operator


class _Omitted:
    def __repr__(self):
        return '<omitted>'


_OMITTED = _Omitted()


class _DefaultDimension:
    def __repr__(self):
        return '0'


_DEFAULT_DIM = _DefaultDimension()
_BOOL_DTYPE = np.dtype('bool')
_BYTE_DTYPE = np.dtype('uint8')


def _empty_product(input, mat2, shape, dtype, aliases=(True, True)):
    if isinstance(input, _autograd.ForwardTensor) or isinstance(mat2, _autograd.ForwardTensor):
        a, da = _autograd.forward_parts(input)
        b, db = _autograd.forward_parts(mat2)
        return _autograd.ForwardTensor(_empty_product(a, b, shape, dtype, aliases),
            _empty_product(da, b, shape, dtype, aliases) + _empty_product(a, db, shape, dtype, aliases))
    shape_a, shape_b = tuple(input.shape), tuple(mat2.shape)
    dtype_a, dtype_b = input.dtype, mat2.dtype
    if np.dtype(dtype).kind not in 'fc' and _state._grad_enabled and (input.requires_grad or mat2.requires_grad):
        raise RuntimeError('An integer or boolean matrix result cannot carry gradients')
    value = np.zeros(shape, dtype=dtype)
    if not value.size:value = _layout.contiguous_empty(value)
    if np.dtype(dtype).kind not in 'fc':
        return _tensor.Tensor(value, _metadata_owned=True)
    saved = []
    if input.requires_grad and aliases[1]:saved.append(mat2)
    if mat2.requires_grad and aliases[0]:saved.append(input)
    return _autograd.result(value,
        (input, lambda gradient: np.zeros(shape_a, dtype=dtype_a)),
        (mat2, lambda gradient: np.zeros(shape_b, dtype=dtype_b)), versioned=saved)


def _empty_bmm(input, mat2):
    return _empty_product(input, mat2, (input.shape[0], input.shape[1], mat2.shape[2]), mat2.dtype)


def _matmul_reverse_grad(value):
    # torch.func.jvp hides reverse flags from its callback, while functional
    # reverse-based JVP and HVP evaluate differentiable private leaves.
    return value.requires_grad and (not isinstance(value, _autograd.ForwardTensor)
        or _state._functional_jvp_active or _state._hvp_active)


def _matmul_fold(input, other):
    large, small = (input, other) if input.ndim >= other.ndim else (other, input)
    if large.ndim < 3 or small.ndim > 2:return False
    if _matmul_reverse_grad(small):return True
    if input.ndim == 2:return False
    if not large.numel():return True
    shape, strides = tuple(large.shape), tuple(large.stride())
    if other.ndim > input.ndim:
        shape = (*shape[:-2], shape[-1], shape[-2])
        strides = (*strides[:-2], strides[-1], strides[-2])
    return all(strides[i] == strides[i+1] * shape[i+1] for i in range(large.ndim - 2))


def _matmul_mode(input, other):
    if input.ndim == other.ndim == 1:return 'dot'
    if input.ndim == 2 and other.ndim == 1:return 'mv'
    if max(input.ndim, other.ndim) <= 2:return 'mm'
    if _matmul_fold(input, other):return 'mv' if min(input.ndim, other.ndim) == 1 else 'mm'
    if input.ndim == other.ndim == 3 and input.shape[0] != other.shape[0]:
        if input.shape[0] == 1 and (_matmul_reverse_grad(input) or isinstance(input, _autograd.ForwardTensor)):
            return _matmul_mode(input.squeeze(0), other)
        if other.shape[0] == 1 and (_matmul_reverse_grad(other) or isinstance(other, _autograd.ForwardTensor)):
            return _matmul_mode(input, other.squeeze(0))
    return 'bmm'


def _reshape_alias(shape, strides, target):
    try:_layout.view_strides(tuple(shape), tuple(strides), tuple(target))
    except RuntimeError:return False
    return True


def _matmul_aliases(input, other, batch):
    if max(input.ndim, other.ndim) <= 2:return True, True
    if _matmul_fold(input, other):
        large = input if input.ndim >= other.ndim else other
        shape, strides = tuple(large.shape), tuple(large.stride())
        if other.ndim > input.ndim:
            shape = (*shape[:-2], shape[-1], shape[-2])
            strides = (*strides[:-2], strides[-1], strides[-2])
        alias = _reshape_alias(shape, strides, (math.prod(shape[:-1]), shape[-1]))
        return (alias, True) if input.ndim >= other.ndim else (True, alias)
    if input.ndim == other.ndim == 3 and input.shape[0] != other.shape[0]:
        if input.shape[0] == 1 and (_matmul_reverse_grad(input) or isinstance(input, _autograd.ForwardTensor)):
            return _matmul_aliases(input.squeeze(0), other, batch)
        if other.shape[0] == 1 and (_matmul_reverse_grad(other) or isinstance(other, _autograd.ForwardTensor)):
            return _matmul_aliases(input, other.squeeze(0), batch)
    aliases = []
    for value, right in [(input, False), (other, True)]:
        shape = tuple(batch) + (tuple(value.shape[-2:]) if value.ndim > 1 else ((value.shape[0],) if right else (1, value.shape[0])))
        padding = len(shape) - value.ndim
        source = (1,) * padding + tuple(value.shape)
        strides = (0,) * padding + tuple(value.stride())
        strides = tuple(0 if x != y else step for x, y, step in zip(source, shape, strides))
        target = (math.prod(batch),) + shape[len(batch):]
        aliases.append(_reshape_alias(shape, strides, target))
    return tuple(aliases)


def _matmul_special(input, other):
    # Native 2.6 routes matrix-vector, folded and batched products differently.
    # This boundary uses geometry only; no operand cast/copy or numerical work.
    a, b = input.shape, other.shape
    if not a or not b:raise RuntimeError('matmul requires at least 1D operands')
    if a[-1] != (b[-2] if len(b) > 1 else b[0]):
        raise RuntimeError('matmul contraction dimensions do not match')
    aa, bb = a[:-2] if len(a) > 1 else (), b[:-2] if len(b) > 1 else ()
    size = max(len(aa), len(bb));aa = (1,) * (size-len(aa)) + tuple(aa);bb = (1,) * (size-len(bb)) + tuple(bb)
    batch = []
    for x, y in zip(aa, bb):
        if x == y:batch.append(x)
        elif x == 1:batch.append(y)
        elif y == 1:batch.append(x)
        else:raise RuntimeError('matmul batch dimensions do not match')
    mode = _matmul_mode(input, other)
    empty = not input.numel() or not other.numel()
    if mode in ('dot', 'mm') or not empty:
        if input.dtype != other.dtype:raise RuntimeError('matmul operands must have the same dtype')
        if (input.dtype.kind == 'b' or input.dtype.kind == 'u' and input.dtype.itemsize > 1) and (mode == 'dot' or not empty):
            raise RuntimeError('matmul is not implemented for this dtype')
        dtype = input.dtype
    else:
        dtype = (input.dtype if other.ndim > input.ndim else other.dtype) if mode == 'mv' else other.dtype
    if empty:
        shape = tuple(batch) + (tuple(a[-2:-1]) if len(a) > 1 else ()) + (tuple(b[-1:]) if len(b) > 1 else ())
        records_graph = _state._grad_enabled and (input.requires_grad or other.requires_grad)
        forward = isinstance(input, _autograd.ForwardTensor) or isinstance(other, _autograd.ForwardTensor)
        aliases = _matmul_aliases(input, other, batch) if records_graph or forward else (True, True)
        return shape, dtype, aliases
    return None


def _matrix_product(input, mat2, rank):
    name = 'mm' if rank == 2 else 'bmm'
    if not isinstance(input, _tensor.Tensor) or not isinstance(mat2, _tensor.Tensor):
        raise TypeError(f'{name} operands must be Tensors')
    if input.ndim != rank or mat2.ndim != rank:
        raise RuntimeError(f'{name} operands must have {rank} dimensions')
    if input.shape[-1] != mat2.shape[-2] or rank == 3 and input.shape[0] != mat2.shape[0]:
        raise RuntimeError(f'{name} operand dimensions do not match')
    # Empty bmm needs metadata only. Avoid casting a potentially large
    # nonempty operand when the result is empty, including complex inputs.
    if rank == 3 and (not input.numel() or not mat2.numel()):
        return _empty_bmm(input, mat2)
    if input.dtype != mat2.dtype:
        raise RuntimeError(f'{name} operands must have the same dtype')
    if input.numel() and mat2.numel() and (input.dtype.kind == 'b' or input.dtype.kind == 'u' and input.dtype.itemsize > 1):
        raise RuntimeError(f'{name} is not implemented for this dtype')
    return input @ mat2


def mm(input, mat2, *, out=None):
    if out is not None:
        raise _errors.BrowserUnsupported('Matrix output buffers require the server')
    return _matrix_product(input, mat2, 2)


def bmm(input, mat2, *, out=None):
    if out is not None:
        raise _errors.BrowserUnsupported('Matrix output buffers require the server')
    return _matrix_product(input, mat2, 3)


def _product_tensors(input, other, name):
    if not isinstance(input, _tensor.Tensor) or not isinstance(other, _tensor.Tensor):
        raise TypeError(f'{name} operands must be Tensors')


def matmul(input, other, *, out=None):
    if out is not None:raise _errors.BrowserUnsupported('Matrix output buffers require the server')
    _product_tensors(input, other, 'matmul')
    return input @ other


def dot(input, tensor, *, out=None):
    if out is not None:raise _errors.BrowserUnsupported('Vector output buffers require the server')
    _product_tensors(input, tensor, 'dot')
    if input.ndim != 1 or tensor.ndim != 1:raise RuntimeError('dot requires two vectors')
    return input @ tensor


def mv(input, vec, *, out=None):
    if out is not None:raise _errors.BrowserUnsupported('Vector output buffers require the server')
    _product_tensors(input, vec, 'mv')
    if input.ndim == 0:raise IndexError('mv input has no dimensions')
    if input.ndim != 2 or vec.ndim != 1:raise RuntimeError('mv requires a matrix and a vector')
    return input @ vec


def unbroadcast(value, shape):
    while value.ndim > len(shape):
        value = value.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and value.shape[axis] != 1:
            value = value.sum(axis=axis, keepdims=True)
    return value.reshape(shape)


def wrap(value):
    if isinstance(value, _tensor.Tensor): return value
    # Operator scalars can carry uint64 values before narrowing to the receiving
    # Tensor dtype. Public tensor() construction instead defaults to signed int64.
    if type(value) is int and 2**63 <= value < 2**64:
        return _tensor.Tensor(value, dtype=np.uint64, _metadata_owned=True)
    return tensor(value)


def inherited_options(dtype, options):
    options=dict(options)
    if options.get('dtype') is None: options['dtype']=dtype
    return options


def reduction_axes(dim, ndim):
    if dim is None: return tuple(range(ndim))
    dims = tuple(dim) if isinstance(dim, (list, tuple)) else (dim,)
    if not dims: raise _errors.BrowserUnsupported('Empty reduction dimension lists require the server')
    if any(not isinstance(d, (int, np.integer)) or not -max(ndim, 1) <= d < max(ndim, 1) for d in dims):
        raise IndexError('Reduction dimension out of range')
    axes = tuple(d % ndim for d in dims) if ndim else ()
    if len(set(axes)) != len(axes): raise RuntimeError('Duplicate reduction dimensions')
    return axes


def power(input, exponent, *, out=None):
    if out is not None: raise _errors.BrowserUnsupported('Power output buffers require the server')
    if isinstance(input, _tensor.Tensor): return input ** exponent
    if isinstance(exponent, _tensor.Tensor): return exponent.__rpow__(input)
    raise TypeError('Power requires a tensor operand')


def count_nonzero(input, dim=None):
    axes = None if dim is None else reduction_axes(dim, input.ndim)
    return _tensor.Tensor(np.count_nonzero(input._data, axis=axes), dtype=np.int64)


def where(condition, input=_OMITTED, other=_OMITTED, *, out=_OMITTED):
    if not isinstance(condition, _tensor.Tensor):
        raise TypeError('where condition must be a Tensor')
    if input is _OMITTED and other is _OMITTED:
        if out is not _OMITTED:
            raise TypeError('where(condition) does not accept out')
        if isinstance(condition, _autograd.ForwardTensor):
            raise _errors.BrowserUnsupported('Nonzero indices inside forward AD are unsupported')
        if condition.dtype.kind == 'u' and condition.dtype.itemsize > 1:
            raise _errors.BrowserUnsupported('Nonzero indices for wider unsigned storage are unsupported')
        return tuple(_tensor.Tensor(value.astype(np.int64, copy=False), _metadata_owned=True)
                     for value in np.nonzero(np.atleast_1d(condition._data)))
    if input is _OMITTED or other is _OMITTED:
        raise TypeError('where requires both input and other')
    if out is not _OMITTED and out is not None:
        raise _errors.BrowserUnsupported('where output buffers require the server')
    left_tensor, right_tensor = isinstance(input, _tensor.Tensor), isinstance(other, _tensor.Tensor)
    left_info = None if left_tensor else _promotion.scalar_info(input)
    right_info = None if right_tensor else _promotion.scalar_info(other)
    if condition.dtype != _BOOL_DTYPE:
        if condition.dtype != _BYTE_DTYPE:
            raise RuntimeError('where condition must have boolean dtype')
        if not _state._where_uint8_warning_emitted:
            _state._where_uint8_warning_emitted = True
            warnings.warn('where received a uint8 condition; use boolean conditions instead.', UserWarning, stacklevel=2)
    same_dtype = left_tensor and right_tensor and input.dtype == other.dtype
    dtype = input.dtype if same_dtype else _promotion.operand_dtype(input, other, left_info=left_info, right_info=right_info)
    left = None if left_info is None else _promotion.scalar_array(left_info[0], dtype, parsed=True)
    right = None if right_info is None else _promotion.scalar_array(right_info[0], dtype, parsed=True)
    if isinstance(input, _autograd.ForwardTensor) or isinstance(other, _autograd.ForwardTensor):
        if left_info is not None: input = _tensor.Tensor(left, _metadata_owned=True)
        if right_info is not None: other = _tensor.Tensor(right, _metadata_owned=True)
        pa, ta = _autograd.forward_parts(input); pb, tb = _autograd.forward_parts(other)
        return _autograd.ForwardTensor(where(condition, pa, pb), where(condition, ta, tb))
    if left_tensor: left = input._data if same_dtype else _promotion.array(input, dtype)
    if right_tensor: right = other._data if same_dtype else _promotion.array(other, dtype)
    mask = condition._data
    try:
        data = np.where(mask, left, right)
    except ValueError as error:
        raise RuntimeError('where operands cannot be broadcast together') from error
    edges = []
    if _state._grad_enabled:
        if left_tensor and input.requires_grad:
            shape = input.shape
            edges.append((input, lambda g, shape=shape: unbroadcast(np.where(mask, g, 0), shape)))
        if right_tensor and other.requires_grad:
            shape = other.shape
            edges.append((other, lambda g, shape=shape: unbroadcast(np.where(mask, 0, g), shape)))
    return _autograd.result(data, *edges, saved=(condition,), versioned=(condition,))


def _numpy_input(data, *, warn_readonly=False):
    """Validate imported arrays before a copy or cast can hide their layout/type."""
    dtype = data.dtype
    if dtype.char not in '?bBhHiIlLqQefdFD':
        raise TypeError(f'Unsupported NumPy dtype: {dtype} ({dtype.type.__name__})')
    if not dtype.isnative:
        raise ValueError('NumPy byte order must match the native byte order')
    # Supported element sizes are powers of two. OR preserves a negative sign
    # and collects every misaligned low bit, including singleton/empty layouts.
    strides = reduce(or_, data.strides, 0)
    if strides < 0 or strides & (dtype.itemsize - 1):
        raise ValueError('NumPy strides must be non-negative multiples of the element size')
    if warn_readonly and not data.flags.writeable and not _state._numpy_readonly_warning_emitted:
        _state._numpy_readonly_warning_emitted = True
        warnings.warn('The given NumPy array is not writable; copy it before writing through a Tensor. '
                      'This warning is suppressed for the rest of this runtime.', UserWarning, stacklevel=3)


def tensor(data, dtype=None, *, device=None, requires_grad=False):
    _device.check_device(device)
    if dtype is not None: dtype = _conversion.dtype(dtype)
    if dtype is None:
        kind = type(data)
        # Builtin scalar construction already owns a fresh allocation. Avoid
        # an intermediate array and a second copy on common arithmetic paths.
        if kind is float:
            if _defaults._float_dtype != _defaults._FLOAT:
                with np.errstate(all='ignore'):
                    return _tensor.Tensor(data, dtype=_defaults._float_dtype, requires_grad=requires_grad, _metadata_owned=True)
            if -3.4028234663852886e38 <= data <= 3.4028234663852886e38 or not np.isfinite(data):
                return _tensor.Tensor(data, dtype=np.float32, requires_grad=requires_grad, _metadata_owned=True)
            with np.errstate(all='ignore'):
                return _tensor.Tensor(data, dtype=np.float32, requires_grad=requires_grad, _metadata_owned=True)
        if kind is int:
            if not -(2**63) <= data < 2**63: raise ValueError('Overflow when unpacking long long')
            return _tensor.Tensor(data, dtype=np.int64, requires_grad=requires_grad, _metadata_owned=True)
        if kind is bool:
            return _tensor.Tensor(data, dtype=np.bool_, requires_grad=requires_grad, _metadata_owned=True)
        if isinstance(data, np.generic) and data.dtype != np.dtype('uint64'):
            return _tensor.Tensor(data, requires_grad=requires_grad, _metadata_owned=True)
        if kind in (list, tuple) and 0 < len(data) <= 8 and type(data[0]) is float:
            floats, guard = True, _defaults._float_dtype == _defaults._HALF
            for value in data:
                if type(value) is not float:
                    floats = False
                    break
                if not -3.4028234663852886e38 <= value <= 3.4028234663852886e38:
                    guard = True
            if floats:
                if guard:
                    with np.errstate(all='ignore'):
                        return _tensor.Tensor(data, dtype=_defaults._float_dtype, requires_grad=requires_grad, _metadata_owned=True)
                return _tensor.Tensor(data, dtype=_defaults._float_dtype, requires_grad=requires_grad, _metadata_owned=True)
        if kind in (list, tuple) and 0 < len(data) <= 8 and type(data[0]) in (int, bool):
            scalar_kind = type(data[0])
            homogeneous, overflow = True, False
            for value in data:
                if type(value) is not scalar_kind:
                    homogeneous = False
                    break
                if scalar_kind is int and not -(2**63) <= value < 2**63:
                    overflow = True
            if homogeneous:
                if overflow:
                    raise ValueError('Overflow when unpacking long long')
                return _tensor.Tensor(data, dtype=np.int64 if scalar_kind is int else np.bool_,
                                      requires_grad=requires_grad, _metadata_owned=True)
    if isinstance(data, _tensor.Tensor):
        array = data._data
    elif isinstance(data, np.ndarray):
        _numpy_input(data)
        array = data
    else:
        owned = _construction.array(data, dtype)
        return _tensor.Tensor(owned, requires_grad=requires_grad, _metadata_owned=True)
    # torch.tensor always owns a copy, including NumPy inputs and strided views.
    if dtype is not None and np.dtype(dtype) != array.dtype:
        with np.errstate(all='ignore'): owned = np.array(array, dtype=dtype, copy=True)
    else:
        owned = np.array(array, dtype=dtype, copy=True)
    return _tensor.Tensor(owned, requires_grad=requires_grad, _metadata_owned=True)


def as_tensor(data, dtype=None, *, device=None):
    _device.check_device(device)
    if dtype is not None: dtype = _conversion.dtype(dtype)
    if isinstance(data, _tensor.Tensor):
        return data if dtype is None or np.dtype(dtype) == data.dtype else data.to(dtype=dtype)
    if isinstance(data, np.ndarray):
        _numpy_input(data, warn_readonly=True)
        if dtype is not None and dtype != data.dtype:
            with np.errstate(all='ignore'): return _tensor.Tensor(data, dtype=dtype)
        return _tensor.Tensor(data)
    return tensor(data, dtype=dtype)


def from_numpy(data):
    if not isinstance(data, np.ndarray): raise TypeError('from_numpy expects a NumPy ndarray')
    _numpy_input(data, warn_readonly=True)
    return _tensor.Tensor(data)


def nan_to_num(input, nan=0., posinf=None, neginf=None, *, out=None):
    if out is not None: raise _errors.BrowserUnsupported('nan_to_num output buffers require the server')
    input = wrap(input)
    value = np.nan_to_num(input._data, nan=nan, posinf=posinf, neginf=neginf)
    finite = np.isfinite(input._data)
    return _autograd.result(value, (input, lambda g: g * finite))


def elementwise_extreme(a, b, maximum=False):
    a = wrap(a)
    a._real_only(); wrap(b)._real_only()
    def share(left, right):
        blocked = left < right if maximum else left > right
        return np.where(left == right, .5, np.where(blocked, 0., 1.))
    return a._binary(b, np.maximum if maximum else np.minimum,
                     lambda g,x,y: g * share(x,y), lambda g,x,y: g * share(y,x))


def trapezoid(y, x=None, *, dx=None, dim=-1):
    y = wrap(y)
    if not y.ndim or not isinstance(dim, (int,np.integer)) or not -y.ndim <= dim < y.ndim:
        raise IndexError('trapezoid dimension out of range')
    if y.dtype.kind not in 'fc': raise _errors.BrowserUnsupported('Integer trapezoid inputs require the server')
    dim %= y.ndim
    if x is not None and dx is not None: raise TypeError('Specify x or dx, not both')
    if y.shape[dim] == 0:
        return _tensor.Tensor(np.zeros(y.shape[:dim] + y.shape[dim+1:],dtype=y.dtype))
    if x is None:
        if dx is not None and (not np.isscalar(dx) or np.iscomplexobj(dx)):
            raise _errors.BrowserUnsupported('trapezoid dx must be a real scalar')
        shape=y.shape;step=np.asarray(1 if dx is None else dx,dtype=y.dtype)
        half=np.asarray(.5,dtype=y.dtype)
        with np.errstate(over='ignore',invalid='ignore'):
            value=(y._data.sum(axis=dim)-(np.take(y._data,0,axis=dim)+np.take(y._data,-1,axis=dim))*half)*step
        def vjp(g):
            with np.errstate(over='ignore',invalid='ignore'):
                scaled=g*step
                gradient=np.broadcast_to(np.expand_dims(scaled,dim),shape).copy()
                correction=scaled*half
                edge=[slice(None)]*len(shape);edge[dim]=0;gradient[tuple(edge)]-=correction
                edge[dim]=-1;gradient[tuple(edge)]-=correction
            return gradient
        return _autograd.result(value,(y,vjp),saves_tensors=False)
    left, right = [slice(None)]*y.ndim, [slice(None)]*y.ndim
    left[dim], right[dim] = slice(None,-1), slice(1,None)
    average = (y[tuple(left)] + y[tuple(right)]) / 2
    x = wrap(x)
    if not x.ndim or x.ndim > y.ndim or x.dtype.kind not in 'fc':
        raise _errors.BrowserUnsupported('trapezoid coordinates require floating tensors of supported rank')
    if x.ndim == 1:
        if x.numel() != y.shape[dim]: raise RuntimeError('trapezoid coordinate length does not match samples')
        shape = [1]*y.ndim; shape[dim] = x.numel(); x = x.reshape(shape)
    else: x = x.reshape((1,)*(y.ndim-x.ndim) + x.shape)
    return (average * (x[tuple(right)] - x[tuple(left)])).sum(dim)


def bincount(input, weights=None, minlength=0):
    if weights is not None: raise _errors.BrowserUnsupported('Weighted bincount requires the server')
    if input.ndim != 1 or input.dtype.kind not in 'iu' or np.any(input._data < 0) or type(minlength) is not int or minlength < 0:
        raise RuntimeError('bincount expects a nonnegative integer vector and minlength')
    return _tensor.Tensor(np.bincount(input._data.astype(np.int64), minlength=minlength), dtype=np.int64)


def assert_close(actual, expected, *, rtol=None, atol=None, equal_nan=False, check_device=True, check_dtype=True, **options):
    if options: raise _errors.BrowserUnsupported('Advanced torch.testing options require the server')
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


def einsum(equation, a, b):
    a, b = wrap(a), wrap(b)
    a_data, b_data = a._data, b._data
    inputs, output = equation.replace(' ', '').split('->')
    left, right = inputs.split(',')
    if '.' in equation or len(set(left)) != len(left) or len(set(right)) != len(right):
        raise _errors.BrowserUnsupported('This einsum pattern requires the server')
    return _autograd.result(np.einsum(equation, a_data, b_data),
                  (a, lambda g: unbroadcast(np.einsum(f'{output},{right}->{left}', g, b_data), a_data.shape)),
                  (b, lambda g: unbroadcast(np.einsum(f'{output},{left}->{right}', g, a_data), b_data.shape)))


def _join_dimension(dim, axis):
    if axis is not _OMITTED:
        if dim is not _DEFAULT_DIM: raise TypeError('Specify dim or axis, not both')
        return axis
    return 0 if dim is _DEFAULT_DIM else dim


def cat(tensors, dim=_DEFAULT_DIM, *, out=None, axis=_OMITTED):
    if out is not None: raise _errors.BrowserUnsupported('cat output buffers require the server')
    return concatenate(tensors, _join_dimension(dim, axis))


def stack(tensors, dim=_DEFAULT_DIM, *, out=None, axis=_OMITTED):
    if out is not None: raise _errors.BrowserUnsupported('stack output buffers require the server')
    return concatenate(tensors, _join_dimension(dim, axis), stack=True)


def concatenate(tensors, dim=0, *, stack=False):
    if not isinstance(tensors, (list, tuple)):
        raise TypeError('Expected a list or tuple of Tensors')
    if type(dim) is bool: raise TypeError('Tensor dimension must be an integer, not bool')
    dim = operator.index(dim)
    tensors = tuple(tensors)
    if not tensors: raise RuntimeError('Expected a non-empty tensor sequence')
    dtype, all_same, forward, empty, active = None, True, False, False, False
    for value in tensors:
        if not isinstance(value, _tensor.Tensor):
            raise TypeError('Expected a list or tuple of Tensors')
        forward = forward or isinstance(value, _autograd.ForwardTensor)
        empty = empty or value.shape == (0,)
        active = active or value.requires_grad
        if dtype is None:
            dtype = value.dtype
        elif dtype != value.dtype:
            all_same = False
            dtype = _construction.promote_types(dtype, value.dtype)
        if not stack and value.ndim == 0:
            raise RuntimeError('Zero-dimensional tensors cannot be concatenated')
    if forward:
        parts = [_autograd.forward_parts(v) for v in tensors]
        return _autograd.ForwardTensor(concatenate([v[0] for v in parts], dim, stack=stack), concatenate([v[1] for v in parts], dim, stack=stack))
    values = tensors if stack or not empty else tuple(value for value in tensors if value.shape != (0,))
    rank = (values[0] if values else tensors[0]).ndim + int(stack)
    if not -rank <= dim < rank: raise IndexError('Tensor dimension out of range')
    dim %= rank
    arrays = [value._data for value in values or tensors] if all_same else [_promotion.array(value, dtype) for value in values or tensors]
    try:
        data = np.stack(arrays, axis=dim) if stack else np.concatenate(arrays, axis=dim)
    except ValueError as error:
        raise RuntimeError('Tensor shapes are incompatible for stack/concatenation') from error
    if not active or not _state._grad_enabled:
        return _autograd.result(data, saves_tensors=False)
    edges, offset = [], 0
    for value in tensors:
        if not stack and value.shape == (0,) and values:
            if value.requires_grad and _state._grad_enabled:
                edges.append((value, lambda g: np.empty((0,), dtype=g.dtype)))
            continue
        start, end = offset, offset + (1 if stack else value.shape[dim])
        if not value.requires_grad:
            offset = end
            continue
        def vjp(g, start=start, end=end):
            index = [slice(None)] * rank
            index[dim] = start if stack else slice(start, end)
            return g[tuple(index)]
        edges.append((value, vjp))
        offset = end
    return _autograd.result(data, *edges, saves_tensors=False)


def cov(x, *, correction=1, fweights=None, aweights=None):
    if fweights is not None or aweights is not None:
        raise _errors.BrowserUnsupported('Weighted covariance requires the server')
    if x.ndim > 2 or x.dtype.kind != 'f': raise _errors.BrowserUnsupported('Covariance requires a floating scalar/vector/matrix')
    if not isinstance(correction, int) or correction < 0: raise ValueError('Invalid covariance correction')
    rows = x._data.reshape(1, -1) if x.ndim < 2 else x._data
    count = rows.shape[-1] - correction
    if count <= 0: raise _errors.BrowserUnsupported('Insufficient covariance observations require the server')
    centered = rows - rows.mean(axis=1, keepdims=True)
    value = centered @ centered.T / count
    def vjp(g):
        matrix = np.asarray(g).reshape(value.shape)
        grad = (matrix + matrix.T) @ centered / count
        return (grad - grad.mean(axis=1, keepdims=True)).reshape(x.shape)
    return _autograd.result(value.squeeze(), (x, vjp))


def _factory_number(value):
    if isinstance(value, _tensor.Tensor):
        if value.ndim:raise TypeError('Factory endpoints and fill values must be scalars')
        value = value.item()
    # Parse a complex Number without asking for inferred complex-half storage.
    if isinstance(value, complex):return complex(value), np.dtype('complex64')
    return _promotion.scalar_info(value)


def eye(n,m=None,*,dtype=None,device=None,requires_grad=False):
    _device.check_device(device)
    if dtype is None:dtype = _defaults._float_dtype
    value = np.eye(n,m,dtype=dtype)
    if not value.size: value = _layout.contiguous_empty(value)
    return _tensor.Tensor(value,requires_grad=requires_grad)


def full(shape, fill_value, *, dtype=None, device=None, requires_grad=False):
    _device.check_device(device)
    fill_value,kind = _factory_number(fill_value)
    if dtype is None:
        # full uses complex64 for a complex Number with default float16.
        dtype = (np.dtype('complex128') if _defaults._float_dtype==np.dtype('float64') else np.dtype('complex64')) if kind.kind=='c' else kind
    value = np.full(shape, fill_value, dtype=dtype)
    if not value.size: value = _layout.contiguous_empty(value)
    return _tensor.Tensor(value, requires_grad=requires_grad)


def full_like(input, fill_value, *, dtype=None, device=None, requires_grad=False, **options):
    if options: raise _errors.BrowserUnsupported('These full_like options require the server')
    _device.check_device(device)
    return _tensor.Tensor(np.full_like(input._data,fill_value,dtype=input.dtype if dtype is None else dtype),requires_grad=requires_grad)


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
    if any(not isinstance(x, _tensor.Tensor) or x.ndim != 1 or x.dtype != tensors[0].dtype for x in tensors):
        raise RuntimeError('cartesian_prod expects one-dimensional tensors with the same dtype')
    if len(tensors) == 1: return tensors[0]
    grids = meshgrid(*tensors, indexing='ij')
    return concatenate([grid.reshape(-1) for grid in grids], dim=1, stack=True)


def unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None):
    input._real_only()
    if input.requires_grad and _state._grad_enabled: raise _errors.BrowserUnsupported('Unique autograd requires the server')
    if dim is not None:
        if not input.ndim or not isinstance(dim,(int,np.integer)) or not -input.ndim <= dim < input.ndim:
            raise IndexError('unique dimension out of range')
        if any(size==0 for size in input.shape): raise _errors.BrowserUnsupported('Empty dimension-wise unique requires the server')
    values,inverse,counts=np.unique(input._data,return_inverse=True,return_counts=True,axis=dim,equal_nan=False)
    outputs=[_tensor.Tensor(values)]
    if return_inverse:outputs.append(_tensor.Tensor(inverse.reshape(input.shape if dim is None else (input.shape[dim],)),dtype=np.int64))
    if return_counts:outputs.append(_tensor.Tensor(counts,dtype=np.int64))
    return outputs[0] if len(outputs)==1 else tuple(outputs)


# Late module references break Tensor/autograd/operator cycles without rebinding functions.
from . import _layout, _autograd, _construction, _conversion, _defaults, _device, _errors, _promotion, _state, _tensor


def permute(input,dims):
    if not isinstance(input,_tensor.Tensor):raise TypeError('permute input must be a Tensor')
    if not isinstance(dims,(tuple,list)):raise TypeError('permute dims must be a sequence')
    return input.permute(dims)


def broadcast_to(input,shape):
    if not isinstance(input,_tensor.Tensor):raise TypeError('broadcast_to input must be a Tensor')
    if not isinstance(shape,(tuple,list)):raise TypeError('broadcast shape must be a sequence')
    return input.expand(shape)


def flatten(input,start_dim=0,end_dim=-1):
    if not isinstance(input,_tensor.Tensor):raise TypeError('flatten input must be a Tensor')
    return input.flatten(start_dim,end_dim)


def transpose(input,dim0,dim1):
    if not isinstance(input,_tensor.Tensor):raise TypeError('transpose input must be a Tensor')
    return input.transpose(dim0,dim1)


def t(input):
    if not isinstance(input,_tensor.Tensor):raise TypeError('t input must be a Tensor')
    return input.t()


def swapaxes(input,axis0,axis1):
    if not isinstance(input,_tensor.Tensor):raise TypeError('swapaxes input must be a Tensor')
    return input.swapaxes(axis0,axis1)


def swapdims(input,dim0,dim1):
    if not isinstance(input,_tensor.Tensor):raise TypeError('swapdims input must be a Tensor')
    return input.swapdims(dim0,dim1)


def movedim(input,source,destination):
    if not isinstance(input,_tensor.Tensor):raise TypeError('movedim input must be a Tensor')
    return input.movedim(source,destination)


def moveaxis(input,source,destination):
    if not isinstance(input,_tensor.Tensor):raise TypeError('moveaxis input must be a Tensor')
    return input.moveaxis(source,destination)


def unsqueeze(input,dim):
    if not isinstance(input,_tensor.Tensor):raise TypeError('unsqueeze input must be a Tensor')
    return input.unsqueeze(dim)


def squeeze(input,dim=_layout.UNSPECIFIED_DIM):
    if not isinstance(input,_tensor.Tensor):raise TypeError('squeeze input must be a Tensor')
    return input.squeeze() if dim is _layout.UNSPECIFIED_DIM else input.squeeze(dim)


from ._sequences import arange, linspace, logspace

from ._distance import cdist, pdist

"""NumPy-backed ops implementation; extracted from PaperLeet."""
import numpy as np
import warnings
from functools import reduce
from operator import or_


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


def where(condition, x, y):
    if isinstance(x, _autograd.ForwardTensor) or isinstance(y, _autograd.ForwardTensor):
        px, tx = _autograd.forward_parts(x); py, ty = _autograd.forward_parts(y)
        return _autograd.ForwardTensor(where(condition, px, py), where(condition, tx, ty))
    condition, a, b = wrap(condition), wrap(x), wrap(y)
    mask = condition._data
    return _autograd.result(np.where(mask, a._data, b._data),
                  (a, lambda g: unbroadcast(np.where(mask, g, 0), a.shape)),
                  (b, lambda g: unbroadcast(np.where(mask, 0, g), b.shape)), saved=(condition,))


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
            floats, guard = True, False
            for value in data:
                if type(value) is not float:
                    floats = False
                    break
                if not -3.4028234663852886e38 <= value <= 3.4028234663852886e38:
                    guard = True
            if floats:
                if guard:
                    with np.errstate(all='ignore'):
                        return _tensor.Tensor(data, dtype=np.float32, requires_grad=requires_grad, _metadata_owned=True)
                return _tensor.Tensor(data, dtype=np.float32, requires_grad=requires_grad, _metadata_owned=True)
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


def concatenate(tensors, dim=0, *, stack=False):
    tensors = [wrap(value) for value in tensors]
    if any(isinstance(v, _autograd.ForwardTensor) for v in tensors):
        parts = [_autograd.forward_parts(v) for v in tensors]
        return _autograd.ForwardTensor(concatenate([v[0] for v in parts], dim, stack=stack), concatenate([v[1] for v in parts], dim, stack=stack))
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


def full(shape, fill_value, *, dtype=None, device=None, requires_grad=False):
    _device.check_device(device)
    if dtype is None: dtype = np.bool_ if isinstance(fill_value, bool) else np.float32 if isinstance(fill_value, float) else np.int64
    return _tensor.Tensor(np.full(shape, fill_value, dtype=dtype), requires_grad=requires_grad)


def full_like(input, fill_value, *, dtype=None, device=None, requires_grad=False, **options):
    if options: raise _errors.BrowserUnsupported('These full_like options require the server')
    _device.check_device(device)
    return _tensor.Tensor(np.full_like(input._data,fill_value,dtype=input.dtype if dtype is None else dtype),requires_grad=requires_grad)


def cdist(x1, x2, p=2., compute_mode='use_mm_for_euclid_dist_if_necessary'):
    if p not in (1, 2) or compute_mode not in ('use_mm_for_euclid_dist_if_necessary', 'use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist'):
        raise _errors.BrowserUnsupported('Only L1 and Euclidean cdist are supported')
    if x1.ndim < 2 or x2.ndim < 2 or x1.shape[-1] != x2.shape[-1] or x1.dtype != x2.dtype:
        raise RuntimeError('cdist expects matching floating feature dimensions and dtypes')
    if x1.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise _errors.BrowserUnsupported('cdist requires float32/float64')
    batch = np.broadcast_shapes(x1.shape[:-2], x2.shape[:-2])
    size = int(np.prod(batch)) * x1.shape[-2] * x2.shape[-2] * max(1, x1.shape[-1])
    if size > 16_000_000: raise _errors.BrowserUnsupported('Large pairwise distances require the server')
    # Direct differences avoid the cancellation of ||a||²+||b||²-2a·b near ties.
    difference = x1._data[..., :, None, :] - x2._data[..., None, :, :]
    lengths = np.abs(difference).sum(axis=-1) if p == 1 else np.linalg.norm(difference, axis=-1)
    def direction(g):
        # Native L1 chooses zero at equal coordinates and propagates NaN signs.
        unit = np.sign(difference) if p == 1 else np.divide(difference, lengths[..., None], out=np.zeros_like(difference), where=lengths[..., None] != 0)
        return g[..., None] * unit
    return _autograd.result(lengths, (x1, lambda g: unbroadcast(direction(g).sum(axis=-2), x1.shape)),
                  (x2, lambda g: unbroadcast(-direction(g).sum(axis=-3), x2.shape)))


def pdist(x, p=2):
    if p != 2 or x.ndim != 2: raise _errors.BrowserUnsupported('Only 2D Euclidean pdist is supported')
    i,j = np.triu_indices(x.shape[0],k=1)
    source = x._data
    difference = source[i] - source[j]
    lengths = np.linalg.norm(difference,axis=1)
    def vjp(g):
        if x.dtype != lengths.dtype:
            raise RuntimeError('pdist backward input and gradient dtypes must match')
        grad = np.zeros_like(source)
        direction = np.divide(difference,lengths[:,None],out=np.zeros_like(difference),where=lengths[:,None]!=0)
        np.add.at(grad,i,g[:,None]*direction);np.add.at(grad,j,-g[:,None]*direction)
        return grad
    return _autograd.result(lengths,(x,vjp))


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
from . import _autograd, _construction, _conversion, _device, _errors, _state, _tensor

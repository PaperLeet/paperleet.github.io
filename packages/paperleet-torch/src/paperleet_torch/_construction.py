"""Dtype inference and scalar storage for sequence-based tensor construction."""
import math
import operator
from collections.abc import Mapping
from itertools import chain

import numpy as np

from . import _conversion


_BOOL = np.dtype('bool')
_LONG = np.dtype('int64')
_FLOAT = np.dtype('float32')
_COMPLEX = np.dtype('complex64')
_DOUBLE_COMPLEX = np.dtype('complex128')
_SCALARS = {bool: _BOOL, int: _LONG, float: _FLOAT, complex: _COMPLEX}
_SCALARS.update({np.dtype(char).type: np.dtype(char) for char in '?bBhHiIlLqQefdFD'})


def promote_types(type1, type2):
    """Promote supported storage dtypes using Torch's category/width rules."""
    if type1 is None or type2 is None:
        raise TypeError('promote_types expects two dtypes')
    left, right = _conversion.dtype(type1), _conversion.dtype(type2)
    if left == right:
        return left
    if left.kind != 'c' and right.kind != 'c':
        floating = [value for value in (left, right) if value.kind == 'f']
        if floating:
            return max(floating, key=lambda value: value.itemsize)
    if left.kind == 'u' and left.itemsize > 1 or right.kind == 'u' and right.itemsize > 1:
        raise RuntimeError('Promotion of uint16, uint32 and uint64 with other dtypes is unsupported')
    if left.kind == 'c' or right.kind == 'c':
        wide = left == _DOUBLE_COMPLEX or right == _DOUBLE_COMPLEX
        wide = wide or left == np.dtype('float64') or right == np.dtype('float64')
        return _DOUBLE_COMPLEX if wide else _COMPLEX
    return np.promote_types(left, right)


def _sequence(value):
    # Tensor implements indexing but is a scalar leaf in this constructor.
    return (not isinstance(value, (_tensor.Tensor, Mapping, np.generic))
            and hasattr(type(value), '__getitem__'))


def _shape(data):
    shape = []
    while _sequence(data):
        size = len(data)  # A first-position zero-dimensional ndarray is unsized.
        shape.append(size)
        if len(shape) > 128:
            raise ValueError('Too many dimensions in tensor input')
        if size == 0:
            break
        data = data[0]
    return tuple(shape)


def _infer(data, active):
    result = _SCALARS.get(type(data))
    if result is not None:
        return result
    if isinstance(data, (np.ndarray, np.generic)):
        return _conversion.dtype(data.dtype).newbyteorder('=')
    if isinstance(data, _tensor.Tensor):
        return data.dtype
    if isinstance(data, (float, int, complex)):
        return _FLOAT if isinstance(data, float) else _LONG if isinstance(data, int) else _COMPLEX
    if isinstance(data, (str, bytes)):
        raise TypeError('Strings cannot be tensor data')
    if not _sequence(data):
        raise RuntimeError('Could not infer dtype of ' + type(data).__name__)
    if id(data) in active:
        raise TypeError('Self-referential sequences are incompatible')
    active.add(id(data))
    try:
        if not len(data):
            return _FLOAT
        result, previous_type, previous_dtype = None, None, None
        for value in data:
            kind = type(value)
            # Repeated primitive scalar types are common in large flat inputs.
            # Containers are always inspected; their contents may differ.
            if kind is previous_type:
                current = previous_dtype
            else:
                current = _SCALARS.get(kind)
                if current is not None:
                    previous_type, previous_dtype = kind, current
                else:
                    current = _infer(value, active)
            result = current if result is None else result if result == current else promote_types(result, current)
            # Native construction stops inference at this maximal dtype. Storage
            # conversion still checks every later element in _store below.
            if result == _DOUBLE_COMPLEX:
                return result
        return result
    finally:
        active.remove(id(data))


def _scalar(value, dtype):
    if isinstance(value, (str, bytes)):
        raise TypeError('Strings cannot be tensor data')
    kind = dtype.kind
    if dtype == np.dtype('uint64'):
        if not isinstance(value, int):
            raise TypeError('An integer is required for uint64 scalar storage')
        if value < 0 or value >= 2**64:
            raise OverflowError('Integer is outside uint64 storage range')
        return value
    if kind in 'iu':
        floating = isinstance(value, float)
        if floating:
            if not math.isfinite(value):
                raise RuntimeError('Non-finite value cannot be converted to an integer without overflow')
            number = math.trunc(value)
        else:
            number = operator.index(value)
            if not -(2**63) <= number < 2**63:
                raise ValueError('Overflow when unpacking long long')
        bits = dtype.itemsize * 8
        if kind == 'u':
            maximum = 2**bits - 1
            overflow = value < 0 if floating else number < -maximum
            overflow = overflow or number > maximum
            number %= 2**bits
        else:
            overflow = not -(2**(bits-1)) <= number < 2**(bits-1)
        if overflow:
            raise RuntimeError('Value cannot be converted to ' + str(dtype) + ' without overflow')
        return number
    if kind == 'b':
        if isinstance(value, (int, float, complex, np.bool_)):
            return bool(value)
        return bool(operator.index(value))
    if isinstance(value, _tensor.Tensor):
        value = value.item()
    return float(value) if kind == 'f' else complex(value)


def _store(data, shape, dtype, level=0):
    if level == len(shape):
        return _scalar(data, dtype)
    if not _sequence(data):
        raise TypeError('Tensor input is not a sequence')
    if len(data) != shape[level]:
        raise ValueError('Inconsistent sequence lengths in tensor input')
    return [_store(value, shape, dtype, level + 1) for value in data]


def _flat(data, dtype):
    """Batch primitive vectors/rows without per-element Python conversion."""
    types = set(map(type, data))
    if types and types <= {list, tuple}:
        types = set(map(type, chain.from_iterable(data)))
    if not types or not types <= _SCALARS.keys():
        return None
    if dtype is None:
        dtypes = {_SCALARS[kind] for kind in types}
        if len(dtypes) > 1 and any(value.kind == 'u' and value.itemsize > 1 for value in dtypes):
            # Wider unsigned promotion can be order-dependent after a floating
            # or maximal-complex element. Keep the ordered inference path.
            return None
        dtype = next(iter(dtypes))
        for other in dtypes:
            if dtype != other:
                dtype = promote_types(dtype, other)
    kind = dtype.kind
    if kind in 'fc':
        if kind == 'f' and complex in types:
            return None
        return np.array(data, dtype=dtype)
    if kind == 'b':
        if not all(issubclass(value, (int, float, complex, np.bool_, np.integer)) for value in types):
            return None
        return np.array(data, dtype=dtype)
    if dtype == np.dtype('uint64') and not types <= {int, bool}:
        return None
    integral = all(issubclass(value, (int, np.integer)) for value in types)
    floating = all(issubclass(value, float) for value in types)
    if not integral and not floating:
        return None
    source = np.asarray(data)
    bits = dtype.itemsize * 8
    if integral:
        if source.dtype.kind not in 'biu':
            return None
        low, high = int(source.min()), int(source.max())
        if dtype == np.dtype('uint64'):
            if low < 0 or high >= 2**64:
                raise OverflowError('Integer is outside uint64 storage range')
        elif low < -(2**63) or high >= 2**63:
            raise ValueError('Overflow when unpacking long long')
        elif kind == 'u':
            if low < -(2**bits-1) or high >= 2**bits:
                raise RuntimeError('Value cannot be converted to ' + str(dtype) + ' without overflow')
        elif low < -(2**(bits-1)) or high >= 2**(bits-1):
            raise RuntimeError('Value cannot be converted to ' + str(dtype) + ' without overflow')
    else:
        truncated = np.trunc(source)
        minimum = 0. if kind == 'u' else -float(2**(bits-1))
        maximum = float(2**bits if kind == 'u' else 2**(bits-1))
        if (not np.isfinite(source).all() or (source < 0).any() and kind == 'u'
                or (truncated < minimum).any() or (truncated >= maximum).any()):
            raise RuntimeError('Value cannot be converted to ' + str(dtype) + ' without overflow')
    # np.asarray(list/tuple) owns this fresh allocation; no caller data aliases it.
    return source if source.dtype == dtype else source.astype(dtype)


def array(data, dtype=None):
    """Create one owned array, retaining NumPy scalar precision in sequences."""
    if isinstance(data, (str, bytes)):
        raise TypeError('Strings cannot be tensor data')
    if type(data) in (list, tuple):
        with np.errstate(all='ignore'):
            result = _flat(data, dtype)
        if result is not None:
            return _layout.contiguous_empty(result) if not result.size else result
    shape = _shape(data)
    inferred = dtype is None
    if inferred:
        dtype = _infer(data, set())
    # Inferred ordinary numeric inputs already have a common valid dtype.
    # Integer/explicit/maximal-complex storage needs scalar protocol/range checks;
    # those are different from an ndarray-to-ndarray cast.
    if not inferred or dtype.kind in 'iu' or dtype == _DOUBLE_COMPLEX:
        data = _store(data, shape, dtype)
    with np.errstate(all='ignore'):
        try:
            result = np.array(data, dtype=dtype, copy=True)
        except (TypeError, ValueError):
            if not inferred:
                raise
            result = np.array(_store(data, shape, dtype), dtype=dtype, copy=True)
        if result.shape != shape:
            result = np.array(_store(data, shape, dtype), dtype=dtype, copy=True)
        return _layout.contiguous_empty(result) if not result.size else result


# Keep a module reference across the Tensor/operator import cycle, as _ops does.
from . import _layout, _tensor

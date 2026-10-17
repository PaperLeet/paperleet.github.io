"""Torch operand categories: dimensioned, zero-dimensional and wrapped scalars."""
import math
import operator

import numpy as np

from . import _construction, _defaults, _errors, _tensor


_BOOL = np.dtype('bool')
_LONG = np.dtype('int64')
_ULONG = np.dtype('uint64')
_FLOAT = np.dtype('float32')
_COMPLEX = np.dtype('complex64')
_DOUBLE = np.dtype('float64')


def is_scalar(value):
    if isinstance(value, np.generic):
        return value.dtype.kind in 'biufc'
    return isinstance(value, (bool, int, float, complex))


def scalar_info(value):
    """Parse a Number argument, distinct from sequence constructor storage."""
    if type(value) is bool:
        return value, _BOOL
    if type(value) is float:
        return value, _defaults._float_dtype
    if type(value) is complex:
        return value, _defaults.scalar_complex_dtype()
    if isinstance(value, (int, np.integer)):
        number = operator.index(value)
        if -(2**63) <= number < 2**63:
            return number, _LONG
        if not 0 <= number < 2**64:
            raise OverflowError('Integer is too large for a scalar argument')
        if not isinstance(value, int):
            raise TypeError('An integer is required for uint64 scalar arguments')
        return number, _ULONG
    if isinstance(value, complex):
        return complex(value), _defaults.scalar_complex_dtype()
    # NumPy complex64 (unlike complex128) is not a Python complex subtype.
    # Native Number argument parsing uses its float protocol, including the
    # usual ComplexWarning. Tensor/sequence construction is a different path.
    if isinstance(value, (float, np.floating, np.bool_, np.complexfloating)):
        return float(value), _defaults._float_dtype
    raise TypeError('Expected a Tensor or numeric scalar')


def _combine(higher, lower, *, wrapped=False):
    if higher.kind == 'c':
        return higher
    if lower.kind == 'c':
        if higher.kind == 'f':
            if higher.itemsize == 2:
                raise _errors.BrowserUnsupported('Complex float16 storage is unsupported by NumPy')
            return np.dtype('complex128') if higher.itemsize == 8 else _COMPLEX
        if wrapped and _defaults._float_dtype==np.dtype('float16'):
            _defaults.complex_dtype()
        return lower
    if higher.kind == 'f':
        return higher
    if higher.kind == 'b' or lower.kind == 'f':
        return _construction.promote_types(higher, lower)
    return higher


def operand_dtype(left, right, *, left_info=None, right_info=None):
    a, b = isinstance(left, _tensor.Tensor), isinstance(right, _tensor.Tensor)
    if a and b:
        if left.dtype == right.dtype:
            return left.dtype
        if bool(left.ndim) == bool(right.ndim):
            return _construction.promote_types(left.dtype, right.dtype)
        return _combine(left.dtype, right.dtype) if left.ndim else _combine(right.dtype, left.dtype)
    if a:
        return _combine(left.dtype, (right_info if right_info is not None else scalar_info(right))[1], wrapped=True)
    if b:
        return _combine(right.dtype, (left_info if left_info is not None else scalar_info(left))[1], wrapped=True)
    left_dtype=(left_info if left_info is not None else scalar_info(left))[1]
    right_dtype=(right_info if right_info is not None else scalar_info(right))[1]
    if (left_dtype.kind=='c' or right_dtype.kind=='c') and _defaults._float_dtype==np.dtype('float16'):
        _defaults.complex_dtype()
    return _construction.promote_types(left_dtype,right_dtype)


def result_type(*args, **kwargs):
    # Native overloads use different keyword names for Tensor/Number pairs.
    signatures = ((('tensor', 'other'), (True, True)),
                  (('scalar', 'tensor'), (False, True)),
                  (('tensor', 'other'), (True, False)),
                  (('scalar1', 'scalar2'), (False, False)))
    if len(args) <= 2:
        for names, tensor_flags in signatures:
            if any(key not in names or names.index(key) < len(args) for key in kwargs):
                continue
            values = list(args)
            if any(name not in kwargs for name in names[len(args):]):
                continue
            values.extend(kwargs[name] for name in names[len(args):])
            if all(isinstance(value, _tensor.Tensor) if flag else is_scalar(value)
                   for value, flag in zip(values, tensor_flags)):
                return operand_dtype(*values)
    raise TypeError('result_type received an invalid combination of Tensor/Number arguments')


def scalar_array(value, dtype, *, parsed=False):
    number = value if parsed else scalar_info(value)[0]
    if dtype.kind in 'iu':
        bits = dtype.itemsize * 8
        if dtype.kind == 'u':
            maximum = 2**bits - 1
            valid = -maximum <= number <= maximum
            number %= 2**bits
        else:
            valid = -(2**(bits-1)) <= number < 2**(bits-1)
        if not valid:
            raise RuntimeError('Scalar cannot be converted to ' + str(dtype) + ' without overflow')
    elif dtype == _FLOAT:
        if math.isfinite(number) and abs(number) > 3.4028234663852886e38:
            raise RuntimeError('Scalar cannot be converted to ' + str(dtype) + ' without overflow')
        # Normal/zero/non-finite double -> float32 conversion cannot underflow.
        if not number or not math.isfinite(number) or abs(number) >= 1.1754943508222875e-38:
            return np.asarray(number, dtype=dtype)
    elif dtype == _DOUBLE:
        return np.asarray(number, dtype=dtype)
    elif dtype == _COMPLEX:
        parts = (number.real, number.imag) if isinstance(number, complex) else (number,)
        if any(math.isfinite(part) and abs(part) > 3.4028234663852886e38 for part in parts):
            raise RuntimeError('Scalar cannot be converted to ' + str(dtype) + ' without overflow')
    with np.errstate(all='ignore'):
        return np.asarray(number, dtype=dtype)


def array(value, dtype):
    if not isinstance(value, _tensor.Tensor):
        return scalar_array(value, dtype)
    if value.dtype == dtype:
        return value._data
    with np.errstate(all='ignore'):
        return value._data.astype(dtype, copy=False)


def sequence_dtype(values):
    dtype = values[0].dtype
    for value in values[1:]:
        if value.dtype != dtype:
            dtype = _construction.promote_types(dtype, value.dtype)
    return dtype

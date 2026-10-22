"""Floating dtype policy shared by construction, promotion and initialization."""
import numpy as np

from . import _errors


_HALF = np.dtype('float16')
_FLOAT = np.dtype('float32')
_DOUBLE = np.dtype('float64')
_COMPLEX = np.dtype('complex64')
_DOUBLE_COMPLEX = np.dtype('complex128')
_float_dtype = _FLOAT


def get_default_dtype():
    return _float_dtype


def set_default_dtype(d):
    if not isinstance(d, np.dtype) or d not in (_HALF, _FLOAT, _DOUBLE):
        raise TypeError('Only float16, float32 or float64 dtypes can be the default')
    global _float_dtype
    _float_dtype = d


def complex_dtype():
    if _float_dtype == _HALF:
        raise _errors.BrowserUnsupported('Default float16 implies complex32 storage, which NumPy cannot represent')
    return _DOUBLE_COMPLEX if _float_dtype == _DOUBLE else _COMPLEX


def scalar_complex_dtype():
    # Wrapped scalar categories may be absorbed by an explicit float32/64 or
    # complex Tensor. Reject complex-half only if it is the resulting storage.
    return _DOUBLE_COMPLEX if _float_dtype == _DOUBLE else _COMPLEX


def step_dtype():
    # Native unfused CPU Adam stores its scalar step in float32 except when
    # the current default is float64, independently of the parameter dtype.
    return _DOUBLE if _float_dtype == _DOUBLE else _FLOAT

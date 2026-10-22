"""Readonly Torch-style numeric limits, independent of Tensor storage and AD."""
from functools import lru_cache

import numpy as np

from . import _defaults


_FLOATS = tuple(np.dtype(name) for name in ('float16', 'float32', 'float64',
                                          'complex64', 'complex128'))
_INTS = tuple(np.dtype(name) for name in ('uint8', 'int8', 'int16', 'int32',
                                        'int64', 'uint16', 'uint32', 'uint64'))


def _argument(args, kwargs, *, floating):
    name = 'finfo' if floating else 'iinfo'
    if not args and not kwargs and floating:
        value = _defaults.get_default_dtype()
    else:
        if len(args) > 1 or any(key != 'type' for key in kwargs) or args and 'type' in kwargs:
            raise TypeError(name + ' expects one type dtype')
        if not args and 'type' not in kwargs:
            raise TypeError(name + ' requires a type dtype')
        value = args[0] if args else kwargs['type']
    # Torch's dtype parser accepts these Python classes as fixed-width types,
    # independently of the default floating dtype. Identity checks deliberately
    # exclude NumPy scalar classes and user subclasses.
    if value is float:
        value = np.dtype('float64')
    elif value is complex:
        value = np.dtype('complex128')
    elif value is int:
        value = np.dtype('int64')
    if not isinstance(value, np.dtype):
        raise TypeError(name + ' requires a dtype object')
    # Keep only canonical builtin dtypes. User NumPy dtype metadata may own
    # arbitrary objects; neither the result nor the bounded cache retains it.
    for dtype in _FLOATS if floating else _INTS:
        if value == dtype:
            if dtype.kind == 'c':
                return np.dtype('float32' if dtype.itemsize == 8 else 'float64')
            return dtype
    raise TypeError(name + ' requires a supported ' + ('floating' if floating else 'integer') + ' dtype')


@lru_cache(maxsize=11)
def _properties(name):
    dtype = np.dtype(name)
    if dtype.kind == 'f':
        value = np.finfo(dtype)
        # Native resolution is computed in double precision, rather than
        # widening NumPy's already rounded float16/float32 resolution scalar.
        return (name, int(value.bits), float(value.min), float(value.max),
                float(value.eps), float(value.tiny), 10.0 ** -int(value.precision))
    value = np.iinfo(dtype)
    return name, int(value.bits), int(value.min), int(value.max)


class _TypeInfo:
    __slots__ = ('_values',)
    __hash__ = None

    def __init__(self, *args, **kwargs):
        # Native initialization is performed by __new__; explicit __init__
        # calls do not change an existing limits object.
        pass

    @property
    def dtype(self): return self._values[0]

    @property
    def bits(self): return self._values[1]

    @property
    def min(self): return self._values[2]

    @property
    def max(self): return self._values[3]

    def __setattr__(self, name, value):
        raise AttributeError('torch.' + type(self).__name__ + ' attributes are readonly')

    def __delattr__(self, name):
        raise AttributeError('torch.' + type(self).__name__ + ' attributes are readonly')

    def __eq__(self, other):
        if type(self) is not type(other): return NotImplemented
        return self.dtype == other.dtype

    def __reduce__(self):
        raise TypeError("cannot pickle 'torch." + type(self).__name__ + "' object")

    def __reduce_ex__(self, protocol):
        return self.__reduce__()


class finfo(_TypeInfo):
    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        if cls is not finfo: raise TypeError('finfo requires its own constructor type')
        dtype = _argument(args, kwargs, floating=True)
        result = object.__new__(cls)
        object.__setattr__(result, '_values', _properties(dtype.name))
        return result

    def __init_subclass__(cls, **kwargs):
        raise TypeError('torch.finfo cannot be subclassed')

    @property
    def eps(self): return self._values[4]

    @property
    def tiny(self): return self._values[5]

    smallest_normal = tiny

    @property
    def resolution(self): return self._values[6]

    def __repr__(self):
        return (f'finfo(resolution={self.resolution:g}, min={self.min:g}, max={self.max:g}, '
                f'eps={self.eps:g}, smallest_normal={self.smallest_normal:g}, '
                f'tiny={self.tiny:g}, dtype={self.dtype})')


class iinfo(_TypeInfo):
    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        if cls is not iinfo: raise TypeError('iinfo requires its own constructor type')
        dtype = _argument(args, kwargs, floating=False)
        result = object.__new__(cls)
        object.__setattr__(result, '_values', _properties(dtype.name))
        return result

    def __init_subclass__(cls, **kwargs):
        raise TypeError('torch.iinfo cannot be subclassed')

    def __repr__(self):
        return f'iinfo(min={self.min:g}, max={self.max:g}, dtype={self.dtype})'

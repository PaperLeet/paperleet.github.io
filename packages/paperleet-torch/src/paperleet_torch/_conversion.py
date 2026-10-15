"""CPU dtype/device conversion arguments shared by Tensor and Module."""
from enum import Enum
import numpy as np

from . import _device, _errors


class MemoryFormat(Enum):
    preserve_format = 'preserve_format'
    contiguous_format = 'contiguous_format'
    channels_last = 'channels_last'
    channels_last_3d = 'channels_last_3d'

    def __repr__(self):
        return 'torch.' + self.value


def memory_format(value, *, contiguous=False):
    if value is None:
        return MemoryFormat.contiguous_format if contiguous else MemoryFormat.preserve_format
    if value in (MemoryFormat.channels_last, MemoryFormat.channels_last_3d):
        raise _errors.BrowserUnsupported('Channels-last memory formats are unsupported')
    if value not in (MemoryFormat.preserve_format, MemoryFormat.contiguous_format):
        raise TypeError('memory_format must be a torch memory format')
    if contiguous and value is MemoryFormat.preserve_format:
        raise RuntimeError('contiguous() does not accept preserve_format')
    return value


def dtype(value):
    if value is None:
        return None
    if type(value) is bool:
        return np.dtype('bool')
    result = np.dtype(value)
    if result.char not in '?bBhHiIlLqQefdFD':
        raise TypeError('A supported boolean or numerical dtype is required')
    return result


def parse_to(args, kwargs, *, module=False):
    from ._tensor import Tensor
    options = dict(kwargs)
    allowed = {'device', 'dtype', 'non_blocking', 'memory_format', 'copy', 'other', 'tensor'}
    unknown = set(options) - allowed
    if unknown:
        raise TypeError('Unexpected to() arguments: ' + ', '.join(sorted(unknown)))
    if module and 'copy' in options:
        raise RuntimeError('Module.to() does not accept copy')
    if 'other' in options and 'tensor' in options:
        raise TypeError('to() received duplicate targets')
    if 'tensor' in options:
        options['other'] = options.pop('tensor')
    if 'other' in options:
        if args:
            raise TypeError('to() received duplicate targets')
        args = (options.pop('other'),)
    positional = {}
    if args:
        first, *rest = args
        if isinstance(first, Tensor):
            positional.update(device=first.device, dtype=first.dtype)
            names = ('non_blocking', 'copy')
        elif first is None or isinstance(first, (_device.Device, str)):
            # Retain the adapter's existing NumPy dtype-string convenience.
            if isinstance(first, str) and first in ('float32', 'float64', 'int64', 'int32', 'bool', 'complex64', 'complex128'):
                positional['dtype'] = first
                names = ('non_blocking', 'copy')
            else:
                positional['device'] = first
                names = ('dtype', 'non_blocking', 'copy')
        else:
            positional['dtype'] = first
            names = ('non_blocking', 'copy')
        if len(rest) > len(names):
            raise TypeError('Too many to() arguments')
        positional.update(zip(names, rest))
    if module and 'copy' in positional:
        raise RuntimeError('Module.to() does not accept copy')
    duplicate = set(positional).intersection(options)
    if duplicate:
        raise TypeError('to() received duplicate arguments: ' + ', '.join(sorted(duplicate)))
    options = {**positional, **options}
    device = options.get('device')
    _device.check_device(device)
    device = _device.Device(device) if device is not None else None
    target = dtype(options.get('dtype'))
    non_blocking, copy = options.get('non_blocking', False), options.get('copy', False)
    if type(non_blocking) is not bool or type(copy) is not bool:
        raise TypeError('non_blocking and copy must be bool')
    copy = copy or device is not None and device.index is not None
    return device, target, non_blocking, copy, memory_format(options.get('memory_format'))


def type_dtype(value):
    if isinstance(value, str):
        names = {'FloatTensor': 'float32', 'DoubleTensor': 'float64', 'LongTensor': 'int64',
                 'IntTensor': 'int32', 'ByteTensor': 'uint8', 'BoolTensor': 'bool',
                 'ComplexFloatTensor': 'complex64', 'ComplexDoubleTensor': 'complex128'}
        if value.startswith('torch.cuda.'):
            raise _errors.BrowserUnsupported('CUDA tensor conversion is unsupported')
        value = names.get(value.removeprefix('torch.'), value)
    return dtype(value)

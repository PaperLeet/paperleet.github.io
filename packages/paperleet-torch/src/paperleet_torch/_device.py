"""Device metadata and explicit CPU-only execution policy."""
import re

from . import _errors


class Device:
    __slots__ = ('_type', '_index')

    def __init__(self, name='cpu', index=None):
        if isinstance(name, Device):
            if index is not None:
                raise RuntimeError('A device index cannot be supplied twice')
            self._type, self._index = name.type, name.index
            return
        if not isinstance(name, str):
            raise TypeError('A device requires a type string')
        match = re.fullmatch(r'([a-z][a-z0-9_]*)(?::([0-9]+))?', name)
        if not match:
            raise RuntimeError('Invalid device string: ' + name)
        kind, parsed = match.groups()
        if kind not in ('cpu', 'cuda', 'mps', 'meta', 'xpu'):
            raise _errors.BrowserUnsupported(f'Device type {kind!r} is unsupported')
        if parsed is not None and index is not None:
            raise RuntimeError('A device index cannot be supplied twice')
        if index is not None and type(index) is not int:
            raise TypeError('A device index must be an integer')
        index = int(parsed) if parsed is not None else index
        if index is not None and not 0 <= index <= 127:
            raise RuntimeError('Device index must be between zero and 127')
        self._type, self._index = kind, index

    type = property(lambda self: self._type)
    index = property(lambda self: self._index)

    def __str__(self):
        return self.type + ('' if self.index is None else ':' + str(self.index))

    def __repr__(self):
        index = '' if self.index is None else ', index=' + str(self.index)
        return 'device(type=' + repr(self.type) + index + ')'

    def __hash__(self):
        return hash((self.type, self.index))

    def __eq__(self, other):
        return isinstance(other, Device) and (self.type, self.index) == (other.type, other.index)


def check_device(device):
    if device is None:
        return
    device = device if isinstance(device, Device) else Device(device)
    if device.type != 'cpu':
        raise _errors.BrowserUnsupported('Only CPU/WASM tensor execution is supported')


_CPU = Device('cpu')

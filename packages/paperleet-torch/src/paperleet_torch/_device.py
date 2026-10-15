"""NumPy-backed device implementation; extracted from PaperLeet."""


def check_device(device):
    if device not in (None, 'cpu', 'cpu:0'): raise _errors.BrowserUnsupported('Only CPU/WASM tensors are supported')


class Device(str):
    def __new__(cls, name='cpu'):
        check_device(name)
        return str.__new__(cls, name)
    type = property(lambda self: 'cpu')
    index = property(lambda self: 0 if self.endswith(':0') else None)


# Late module references break Tensor/autograd/operator cycles without rebinding functions.
from . import _errors

"""Lightweight torch-compatible arrays and analytical autodiff on NumPy.

Run unchanged scripts with ``stridetorch script.py``, or call
``install_as_torch()`` in the host before loading code that imports torch.
Importing this package never replaces an installed PyTorch module.
"""
from ._integration import as_torch, create_torch, install_as_torch

__version__ = '0.42.0'
__all__ = ['torch', 'Tensor', 'UnsupportedOperation', 'create_torch', 'as_torch',
           'install_as_torch']
_namespace = None


def __getattr__(name):
    global _namespace
    if name == 'UnsupportedOperation':
        from ._errors import BrowserUnsupported
        return BrowserUnsupported
    if name not in ('torch', 'Tensor'):
        raise AttributeError(name)
    if _namespace is None:
        _namespace = create_torch(name='stridetorch')
    return _namespace if name == 'torch' else getattr(_namespace, name)

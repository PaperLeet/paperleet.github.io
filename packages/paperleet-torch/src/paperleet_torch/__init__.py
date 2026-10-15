"""Lightweight torch-compatible arrays and analytical autodiff on NumPy.

Use ``from paperleet_torch import torch`` for the compatible namespace.
Importing this package never replaces an installed PyTorch module.
"""
from ._integration import as_torch, create_torch

__version__ = '0.1.0'
__all__ = ['torch', 'Tensor', 'UnsupportedOperation', 'create_torch', 'as_torch']
_namespace = None


def __getattr__(name):
    global _namespace
    if name == 'UnsupportedOperation':
        from ._errors import BrowserUnsupported
        return BrowserUnsupported
    if name.startswith('_'):
        raise AttributeError(name)
    if _namespace is None:
        _namespace = create_torch(name='paperleet_torch')
    return _namespace if name == 'torch' else getattr(_namespace, name)

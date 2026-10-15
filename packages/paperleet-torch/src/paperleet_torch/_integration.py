"""Explicit integration with existing code that imports torch."""
import contextlib
import sys


def create_torch(name='torch', *, register=False):
    """Return a compatible namespace; leave imports untouched by default.

    ``register=True`` registers only the namespace's children. Existing
    PaperLeet workers explicitly assign the returned root to sys.modules.
    """
    from ._api import create_torch as factory
    return factory(name, register=register)


def install():
    """Preserve the original adapter's child-module installation contract."""
    return create_torch(register=True)


@contextlib.contextmanager
def as_torch():
    """Temporarily make import torch resolve to this library, then restore it.

    This changes process-wide import state. Use in an isolated worker or
    process; it does not provide thread-local or independent numerical state.
    """
    previous = {name: module for name, module in sys.modules.items()
                if name == 'torch' or name.startswith('torch.')}
    namespace = create_torch()
    try:
        sys.modules.update(namespace._compat_modules)
        yield namespace
    finally:
        for name in list(sys.modules):
            if name == 'torch' or name.startswith('torch.'):
                del sys.modules[name]
        sys.modules.update(previous)

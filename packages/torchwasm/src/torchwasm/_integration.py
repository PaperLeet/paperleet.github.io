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


def _clear_torch_imports():
    for name in list(sys.modules):
        if name == 'torch' or name.startswith('torch.'):
            del sys.modules[name]


def install_as_torch():
    """Enable ordinary torch imports for this process and return the namespace.

    Call this in the host before loading user code. It replaces the root and
    all cached torch submodules; references already held by other code cannot
    be changed. For a temporary alias, use ``as_torch()`` instead.
    """
    namespace = create_torch()
    _clear_torch_imports()
    sys.modules.update(namespace._compat_modules)
    return namespace


@contextlib.contextmanager
def as_torch():
    """Temporarily make import torch resolve to this library, then restore it.

    This changes process-wide import state. Use in an isolated worker or
    process; it does not provide thread-local or independent numerical state.
    """
    previous = {name: module for name, module in sys.modules.items()
                if name == 'torch' or name.startswith('torch.')}
    try:
        namespace = install_as_torch()
        yield namespace
    finally:
        _clear_torch_imports()
        sys.modules.update(previous)

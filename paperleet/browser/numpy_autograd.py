"""Compatibility entry point; implementation lives in the independent paperleet-torch package."""
try:
    from paperleet._vendor.torch import _backend as _implementation
    from paperleet._vendor.torch._backend import *
except ModuleNotFoundError as error:
    if error.name not in ('paperleet._vendor', 'paperleet._vendor.torch'):
        raise
    # Offline browser bundles contain the standalone namespace only.
    from paperleet_torch import _backend as _implementation
    from paperleet_torch._backend import *


def __getattr__(name):
    return getattr(_implementation, name)

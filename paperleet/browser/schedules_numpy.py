"""Compatibility entry point; implementation lives in the independent stridetorch package."""
try:
    from paperleet._vendor.torch import _legacy as _implementation
    from paperleet._vendor.torch._legacy import *
except ModuleNotFoundError as error:
    if error.name not in ('paperleet._vendor', 'paperleet._vendor.torch'):
        raise
    # Offline browser bundles contain the standalone namespace only.
    from stridetorch import _legacy as _implementation
    from stridetorch._legacy import *


def __getattr__(name):
    return getattr(_implementation, name)

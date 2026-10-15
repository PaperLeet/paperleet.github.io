"""Compatibility entry point; implementation lives in the independent paperleet-torch package."""
from paperleet_torch import _backend as _implementation
from paperleet_torch._backend import *


def __getattr__(name):
    return getattr(_implementation, name)

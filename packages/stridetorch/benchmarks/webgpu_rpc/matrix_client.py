"""Compatibility import; canonical implementation ships in stridetorch."""
import sys
from stridetorch.experimental.webgpu import matrix_client as _implementation
sys.modules[__name__] = _implementation

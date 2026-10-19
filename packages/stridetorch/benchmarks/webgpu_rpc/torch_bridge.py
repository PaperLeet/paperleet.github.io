"""Compatibility import; canonical implementation ships in stridetorch."""
import sys
from stridetorch.experimental.webgpu import torch_bridge as _implementation
sys.modules[__name__] = _implementation

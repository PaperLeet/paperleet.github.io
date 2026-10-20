"""Compatibility import; canonical implementation ships in torchwasm."""
import sys
from torchwasm.experimental.webgpu import torch_bridge as _implementation
sys.modules[__name__] = _implementation

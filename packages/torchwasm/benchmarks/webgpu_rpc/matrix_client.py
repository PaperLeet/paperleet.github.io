"""Compatibility import; canonical implementation ships in torchwasm."""
import sys
from torchwasm.experimental.webgpu import matrix_client as _implementation
sys.modules[__name__] = _implementation

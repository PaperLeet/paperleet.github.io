import math
import torch
from torch import Tensor
import torch.nn.functional as F


def _floating(*values):
    first = values[0]
    for value in values:
        if not value.is_floating_point() or value.dtype != first.dtype or value.device != first.device or not torch.isfinite(value).all():
            raise ValueError("floating inputs must be finite and share dtype/device")

def chunked_mrope(x: Tensor, positions: Tensor, inv_freq: Tensor) -> Tensor:
    """x is [N,2R], positions [3,N] ordered time/height/width, and inv_freq [R]. R is a positive multiple of 3. This equal-width teaching configuration assigns the first/middle/last R/3 frequencies to time/height/width. Rotate matching coordinates in the first and second feature halves, not adjacent coordinates. Floating positions are differentiable; finite inputs share dtype/device."""
    raise NotImplementedError("Implement chunked_mrope")

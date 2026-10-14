import math
import torch
from torch import Tensor


def apply_rope(x: Tensor, positions: Tensor, base: float = 10000.) -> Tensor:
    """Required: x [B,H,L,D], positions [L], D positive and even.

    Pair adjacent coordinates (0,1), (2,3), ... and rotate counterclockwise
    by positions[l] * base**(-2*i/D) for pair i. Preserve x shape/dtype/device
    and gradients into x and real-valued positions; do not modify either input.
    Broadcast positions across batch and heads. Signed, nonconsecutive and real
    positions are supported, including cached-decoding offsets. Empty B/L work.
    Raise ValueError for wrong layout/position length or nonfinite or <= 1 base
    (boolean bases are invalid). Inputs are finite; x is float32 or float64.
    """
    raise NotImplementedError()

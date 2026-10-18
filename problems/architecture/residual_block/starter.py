import torch
from torch import Tensor


def residual_block(x: Tensor, w1: Tensor, w2: Tensor,
                   projection: Tensor | None = None) -> Tensor:
    """Return a Tensor of shape [B,D_out], using x [B,D_in], w1 [D_in,H],
    w2 [H,D_out]. Compute ReLU(ReLU(x @ w1) @ w2 + shortcut).
    projection=None uses x and requires D_in=D_out; otherwise use x @ projection,
    with projection [D_in,D_out], including square projections. Raise ValueError
    for incompatible shapes. B may be zero; feature widths are positive.
    Preserve matching float32/float64 dtype/device, all input/weight gradients,
    and input values. No bias, normalization or in-place activation is required."""
    raise NotImplementedError

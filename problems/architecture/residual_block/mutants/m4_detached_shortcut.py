# BUG: Detach the shortcut gradient
import torch
from torch import Tensor


def residual_block(x: Tensor, w1: Tensor, w2: Tensor,
                   projection: Tensor | None = None) -> Tensor:
    """Return ReLU(ReLU(x @ w1) @ w2 + shortcut), without in-place updates."""
    if x.ndim != 2 or w1.ndim != 2 or w2.ndim != 2:
        raise ValueError('x, w1 and w2 must be matrices')
    if x.shape[1] != w1.shape[0] or w1.shape[1] != w2.shape[0]:
        raise ValueError('residual branch dimensions do not match')
    if min(x.shape[1], w1.shape[1], w2.shape[1]) < 1:
        raise ValueError('feature dimensions must be positive')
    if projection is None:
        if x.shape[1] != w2.shape[1]:
            raise ValueError('a projection is required when feature widths differ')
        shortcut = x
    else:
        if projection.ndim != 2 or projection.shape != (x.shape[1], w2.shape[1]):
            raise ValueError('projection must have shape [D_in, D_out]')
        shortcut = x @ projection
    residual = torch.relu(x @ w1) @ w2
    return torch.relu(residual + shortcut.detach())

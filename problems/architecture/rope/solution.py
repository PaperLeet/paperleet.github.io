import math
import torch
from torch import Tensor


def apply_rope(x: Tensor, positions: Tensor, base: float = 10000.) -> Tensor:
    """Rotate adjacent feature pairs of x [B,H,L,D] at positions [L]."""
    if x.ndim != 4 or x.shape[-1] <= 0 or x.shape[-1] % 2:
        raise ValueError('x must be [B,H,L,D] with positive even D')
    if positions.ndim != 1 or positions.shape[0] != x.shape[-2]:
        raise ValueError('positions must be [L]')
    if isinstance(base, bool) or not isinstance(base, (int, float)) or not math.isfinite(base) or base <= 1:
        raise ValueError('base must be finite and greater than one')
    width = x.shape[-1]
    frequency = base ** (-torch.arange(0, width, 2, dtype=x.dtype, device=x.device) / width)
    phase = positions.to(dtype=x.dtype, device=x.device)[None, None, :, None] * frequency
    cosine, sine = phase.cos(), phase.sin()
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).reshape(x.shape)

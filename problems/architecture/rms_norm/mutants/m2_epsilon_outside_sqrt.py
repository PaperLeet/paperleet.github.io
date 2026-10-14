# BUG: Uses an alternate epsilon placement outside this exercise contract.
import torch
from torch import Tensor


def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
    """Feature-local RMS normalization, with epsilon inside the square root."""
    inverse_rms = 1 / (x.square().mean(dim=-1, keepdim=True).sqrt() + eps)
    return x * inverse_rms * weight


def partial_rms_norm(x: Tensor, weight: Tensor, k: int, eps: float = 1e-5) -> Tensor:
    """Estimate RMS from the first k features, then normalize all features."""
    if type(k) is not int or not 1 <= k <= x.shape[-1]:
        raise ValueError('k must be an integer between 1 and the feature width')
    inverse_rms = torch.rsqrt(x[..., :k].square().mean(dim=-1, keepdim=True) + eps)
    return x * inverse_rms * weight

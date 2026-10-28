import math
import torch
from torch import Tensor
import torch.nn.functional as F


def _finite(*xs):
    if any(not x.numel() or not x.is_floating_point() or not torch.isfinite(x).all() for x in xs):
        raise ValueError("nonempty finite floating tensors required")
    if any(x.dtype != xs[0].dtype or x.device != xs[0].device for x in xs):
        raise ValueError("dtype and device must agree")


def _positive(value):
    if not math.isfinite(value) or value <= 0:
        raise ValueError("positive finite scalar required")

def ranking_loss(preferred: Tensor, rejected: Tensor) -> Tensor:
    """Equal nonempty (B,) reward scores; return mean pairwise negative log likelihood."""
    _finite(preferred,rejected)
    if preferred.ndim!=1 or preferred.shape!=rejected.shape: raise ValueError("equal vectors required")
    return F.softplus(rejected-preferred).mean()


def refl_clean(latent: Tensor, epsilon: Tensor, alpha: float, sigma: float) -> Tensor:
    """Equal (B,D) tensors; alpha/sigma are noising amplitudes; freeze latent only."""
    _finite(latent,epsilon); _positive(alpha)
    if latent.ndim!=2 or latent.shape!=epsilon.shape or not math.isfinite(sigma) or sigma<0: raise ValueError("invalid shape or sigma")
    return (latent.detach()-sigma*epsilon)/alpha

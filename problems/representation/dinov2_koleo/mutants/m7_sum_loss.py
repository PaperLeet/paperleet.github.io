# BUG: Sum regularization over the batch.
"""DINOv2's KoLeo entropy regularizer, with the official fixed PairwiseDistance epsilon."""
import math
import torch
from torch import Tensor


def nearest_neighbors(unit_features: Tensor) -> Tensor:
    """[B,D] unit rows, B>=2; nearest non-self cosine neighbor, lowest-index ties."""
    _features(unit_features)
    similarities = unit_features.detach() @ unit_features.detach().T
    mask = torch.eye(unit_features.shape[0], dtype=torch.bool, device=unit_features.device)
    return similarities.masked_fill(mask, float('-inf')).argmax(dim=1)


def koleo_loss(features: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalize rows; -mean(log(||z_i-z_nn(i)+1e-8||_2+eps)), both endpoints differentiable."""
    _features(features)
    if not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be finite and positive')
    unit = features / features.norm(dim=1, keepdim=True).clamp_min(eps)
    indices = nearest_neighbors(unit)
    delta = unit - unit[indices] + 1e-8
    distances = delta.norm(dim=1)
    return -(distances + eps).log().sum()


def _features(x):
    if not isinstance(x, Tensor) or x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError('features must have shape [B,D] with B>=2,D>=1')
    if x.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(x).all()):
        raise ValueError('features must be finite float32/float64')

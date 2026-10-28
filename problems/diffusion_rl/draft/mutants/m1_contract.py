# Backpropagate through reused generated anchor
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

def draft_lv_loss(generated: Tensor, noises: Tensor, alpha: float, sigma: float, denoiser, reward) -> Tensor:
    """Average original and K re-noised reward branches, preserving only intended gradient paths."""
    _finite(generated,noises);_positive(alpha)
    if generated.ndim!=2 or noises.ndim!=3 or noises.shape[1:]!=generated.shape or not math.isfinite(sigma) or sigma<0:raise ValueError("invalid re-noising shapes/amplitudes")
    first=reward(generated)
    if first.shape!=generated.shape[:1]:raise ValueError("reward must return (B,)")
    total=first.mean()
    for noise in noises:
        noisy=alpha*generated+sigma*noise.detach()
        epsilon=denoiser(noisy)
        if epsilon.shape!=generated.shape:raise ValueError("denoiser changed shape")
        clean=(noisy-sigma*epsilon)/alpha
        values=reward(clean)
        if values.shape!=generated.shape[:1]:raise ValueError("reward must return (B,)")
        total=total+values.mean()
    return -total/(len(noises)+1)

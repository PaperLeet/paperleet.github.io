# BUG: Put epsilon outside the square root
import math
import torch
from torch import Tensor


def batch_norm(x: Tensor, weight: Tensor, bias: Tensor, running_mean: Tensor,
               running_var: Tensor, training: bool = True, momentum: float = .1,
               eps: float = 1e-5) -> tuple[Tensor, Tensor, Tensor]:
    """Return (output, next_mean, next_variance); never mutate input buffers."""
    if x.ndim < 2 or x.shape[1] < 1 or any(size < 1 for size in x.shape[2:]):
        raise ValueError('expected [N,C,*spatial] with positive feature dimensions')
    channels = x.shape[1]
    if any(value.shape != (channels,) for value in (weight, bias, running_mean, running_var)):
        raise ValueError('affine parameters and buffers must have shape [C]')
    if type(training) is not bool or not math.isfinite(eps) or eps <= 0:
        raise ValueError('training must be bool and eps must be positive and finite')
    if not math.isfinite(momentum) or not 0 <= momentum <= 1:
        raise ValueError('momentum must lie in [0,1]')
    shape = (1, channels) + (1,) * (x.ndim - 2)
    old_mean, old_var = running_mean.detach(), running_var.detach()
    if training:
        count = x.numel() // channels
        if count < 2:
            raise ValueError('training needs at least two values per channel')
        axes = (0,) + tuple(range(2, x.ndim))
        mean = x.mean(dim=axes, keepdim=True)
        variance = (x - mean).square().mean(dim=axes, keepdim=True)
        next_mean = (1 - momentum) * old_mean + momentum * mean.detach().reshape(channels)
        next_var = (1 - momentum) * old_var + momentum * variance.detach().reshape(channels) * count / (count - 1)
    else:
        mean, variance = old_mean.reshape(shape), old_var.reshape(shape)
        next_mean, next_var = old_mean, old_var
    normalized = (x - mean) / (torch.sqrt(variance) + eps)
    output = normalized * weight.reshape(shape) + bias.reshape(shape)
    return output, next_mean, next_var

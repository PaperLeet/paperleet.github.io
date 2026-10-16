# BUG: 将 epsilon 放在平方根外
import torch
from torch import Tensor


def group_norm(x: Tensor, num_groups: int, weight: Tensor, bias: Tensor,
               eps: float = 1e-5) -> Tensor:
    """Population moments over each sample's contiguous channel group and space."""
    n, channels = x.shape[:2]
    if type(num_groups) is not int or not 1 <= num_groups <= channels or channels % num_groups:
        raise ValueError('num_groups must be a positive integer divisor of C')
    grouped = x.reshape(n, num_groups, channels // num_groups, *x.shape[2:])
    axes = tuple(range(2, grouped.ndim))
    mean = grouped.mean(dim=axes, keepdim=True)
    centered = grouped - mean
    variance = centered.square().mean(dim=axes, keepdim=True)
    normalized = (centered / (variance.sqrt() + eps)).reshape(x.shape)
    affine_shape = (1, channels) + (1,) * (x.ndim - 2)
    return normalized * weight.reshape(affine_shape) + bias.reshape(affine_shape)

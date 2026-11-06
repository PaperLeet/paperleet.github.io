# Bug: Use same-view rather than crossed targets.
"""BYOL regression and parameter EMA, with the official sum-of-directions scale."""
import math
import torch
from torch import Tensor


def byol_loss(p1: Tensor, p2: Tensor, z1: Tensor, z2: Tensor, eps: float = 1e-12) -> Tensor:
    """Sum two mean squared normalized-vector losses; stop both crossed targets."""
    _check_views(p1, p2, z1, z2)
    _positive(eps, 'eps')
    def normalized(value):
        return value / value.square().sum(dim=1, keepdim=True).clamp_min(eps).sqrt()
    first = (normalized(p1) - normalized(z1.detach())).square().sum(dim=1)
    second = (normalized(p2) - normalized(z1.detach())).square().sum(dim=1)
    return (first + second).mean()


def ema_update(online: Tensor, target: Tensor, momentum: float = .99) -> Tensor:
    """Return independent detached target storage: m*old_target+(1-m)*online."""
    _check_pair(online, target)
    _momentum(momentum)
    return (momentum * target.detach() + (1 - momentum) * online.detach()).clone()


def _check_pair(a, b):
    if not isinstance(a, Tensor) or not isinstance(b, Tensor) or a.shape != b.shape or a.numel() == 0:
        raise ValueError('inputs must have matching nonempty shapes')
    if a.dtype not in (torch.float32, torch.float64) or a.dtype != b.dtype or a.device != b.device:
        raise ValueError('inputs must share float32/float64 dtype and device')
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
        raise ValueError('inputs must be finite')


def _check_views(*values):
    for value in values:
        _check_pair(values[0], value)
    if values[0].ndim != 2:
        raise ValueError('views must have shape [B,D]')


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + ' must be a finite positive Python number')


def _momentum(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('momentum must be a finite Python number in [0,1]')

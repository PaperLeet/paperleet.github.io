import math
import torch
from torch import Tensor


def _float(x, ndim=None):
    if not isinstance(x, Tensor) or not x.is_floating_point() or x.numel() == 0 or (ndim is not None and x.ndim != ndim) or not torch.isfinite(x).all():
        raise ValueError("expected a nonempty finite floating tensor with the declared rank")


def _same(a, b):
    _float(b)
    if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
        raise ValueError("tensor shape, dtype and device must match")

def flow_interpolate(actions: Tensor, noise: Tensor, time: Tensor) -> tuple:
    'Official openpi time convention: t=0 actions, t=1 noise; fixed velocity labels.'
    raise NotImplementedError("Implement flow_interpolate according to the README contract")


def euler_step(state: Tensor, velocity: Tensor, dt: float) -> Tensor:
    'One reverse-time action-sampling step with nonpositive scalar dt.'
    raise NotImplementedError("Implement euler_step according to the README contract")

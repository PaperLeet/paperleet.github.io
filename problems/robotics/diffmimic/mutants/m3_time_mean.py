# BUG: State matching averages the time horizon instead of summing it.
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

def replay_state(state: Tensor, reference: Tensor, threshold: float) -> Tensor:
    """Use current state only below the strict squared-error threshold; otherwise replay reference."""
    _float(state,2);_same(state,reference)
    if not math.isfinite(threshold) or threshold<0: raise ValueError("invalid replay threshold")
    keep=(state-reference.detach()).square().sum(dim=-1)<threshold
    return torch.where(keep.unsqueeze(-1),state,reference.detach())


def pd_torque(position: Tensor, velocity: Tensor, target: Tensor, kp: float, kd: float) -> Tensor:
    """PD target-position control with zero target angular velocity."""
    _float(position,2);_same(position,velocity);_same(position,target)
    if not math.isfinite(kp) or not math.isfinite(kd) or kp<0 or kd<0: raise ValueError("invalid PD gains")
    return kp*(target-position)-kd*velocity


def state_matching_loss(states: Tensor, reference: Tensor) -> Tensor:
    """Sum per-trajectory state-matching errors over time/coordinates, then average batch."""
    _float(states,3);_same(states,reference)
    return (states-reference.detach()).square().mean(dim=-1).mean(dim=-1).mean()

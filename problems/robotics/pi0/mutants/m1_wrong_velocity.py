# BUG: Flow labels use the opposite time direction.
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
    """Official openpi time convention: t=0 actions, t=1 noise; fixed velocity labels."""
    _float(actions,3);_same(actions,noise);_float(time,1)
    if time.shape!=actions.shape[:1] or time.dtype!=actions.dtype or time.device!=actions.device or ((time<0)|(time>1)).any(): raise ValueError("invalid flow time")
    t=time[:,None,None]
    return t*noise+(1-t)*actions,(actions-noise).detach()


def euler_step(state: Tensor, velocity: Tensor, dt: float) -> Tensor:
    """One reverse-time action-sampling step with nonpositive scalar dt."""
    _float(state,3);_same(state,velocity)
    if not math.isfinite(dt) or dt>0: raise ValueError("sampling dt must be finite and nonpositive")
    return state+dt*velocity

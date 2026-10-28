# Allow window to run past the trajectory
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

def window_start(update: int, total: int, width: int, interval: int, stride: int) -> int:
    """Zero-based completed-update index; sliding window saturates at the final valid start."""
    if any(type(x) is not int for x in (update,total,width,interval,stride)) or update<0 or not 1<=width<=total or interval<1 or stride<1:raise ValueError("invalid integer window parameters")
    return min((update//interval)*stride,total-1)


def mixed_transition(state: Tensor, velocity: Tensor, time: float, next_time: float, noise_level: float, stochastic: bool) -> tuple[Tensor,Tensor]:
    """Return Gaussian mean/std for SDE steps or deterministic ODE mean with zero std."""
    _finite(state,velocity)
    if state.ndim!=2 or state.shape!=velocity.shape or not 0<=next_time<time<1 or not math.isfinite(noise_level) or noise_level<0 or type(stochastic) is not bool:raise ValueError("invalid mixed transition")
    dt=next_time-time;x=state.detach();sigma2=noise_level**2*time/(1-time) if stochastic else 0.
    drift=velocity+sigma2/(2*time)*(x+(1-time)*velocity)
    return x+dt*drift,state.new_tensor(math.sqrt(sigma2*(-dt)))


def window_loss(log_ratios: Tensor, advantages: Tensor, start: int, width: int, clip: float) -> Tensor:
    """Average clipped policy terms over the selected (B,width) window only."""
    _finite(log_ratios,advantages)
    if log_ratios.ndim!=2 or advantages.shape!=log_ratios.shape[:1] or type(start) is not int or type(width) is not int or start<0 or width<1 or start+width>log_ratios.shape[1] or not 0<clip<1:raise ValueError("invalid objective window")
    selected=log_ratios[:,start:start+width]
    if (selected.detach().abs()>40).any():raise ValueError("selected log ratios exceed domain")
    ratio=selected.exp();adv=advantages.detach()[:,None]
    return -torch.minimum(ratio*adv,ratio.clamp(1-clip,1+clip)*adv).mean()

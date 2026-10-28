# Average action coordinates instead of joint density
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

def gaussian_log_prob(mean: Tensor, actions: Tensor, std: Tensor) -> Tensor:
    """(B,T,D) means/actions and positive (T,) scales; sum log densities over D."""
    _finite(mean,actions,std)
    if mean.ndim!=3 or actions.shape!=mean.shape or std.shape!=mean.shape[1:2] or (std<=0).any():raise ValueError("invalid Gaussian shapes/scales")
    scale=std.detach()[None,:,None]
    return (-.5*((actions.detach()-mean)/scale).square()-scale.log()-.5*math.log(2*math.pi )).mean(-1)


def ddpo_loss(current: Tensor, old: Tensor, advantages: Tensor, clip: float) -> Tensor:
    """(B,T) transition log probabilities and (B,) frozen terminal advantages."""
    _finite(current,old,advantages)
    if current.ndim!=2 or current.shape!=old.shape or advantages.shape!=current.shape[:1] or not 0<clip<1:raise ValueError("invalid PPO inputs")
    log_ratio=current-old.detach()
    if (log_ratio.detach().abs()>40).any():raise ValueError("log ratios must be within +/-40")
    ratio=log_ratio.exp();adv=advantages.detach()[:,None]
    return -torch.minimum(ratio*adv,ratio.clamp(1-clip,1+clip)*adv).mean()

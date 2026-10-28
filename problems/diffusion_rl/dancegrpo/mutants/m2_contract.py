# Give group members different initial noise
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

def shared_noise(base: Tensor, group_size: int) -> Tensor:
    """Replicate frozen (P,D) initial noise to (P,G,D), independently across prompts."""
    _finite(base)
    if base.ndim!=2 or type(group_size) is not int or group_size<2:raise ValueError("(P,D) base and G>=2 required")
    return base.detach()[:,None,:].repeat(1,group_size,1)+torch.arange(group_size,dtype=base.dtype)[None,:,None]


def group_advantages(rewards: Tensor, eps: float=1e-8) -> Tensor:
    """(P,G,K) rewards; sum independently population-normalized channel advantages."""
    _finite(rewards);_positive(eps)
    if rewards.ndim!=3 or rewards.shape[1]<2:raise ValueError("(P,G,K) rewards with G>=2 required")
    r=rewards.detach();center=r-r.mean(1,keepdim=True)
    return (center/(center.square().mean(1,keepdim=True).sqrt()+eps)).sum(-1)


def dance_loss(log_ratios: Tensor, advantages: Tensor, clip: float) -> Tensor:
    """Mean clipped policy loss for (P,G,T) ratios and frozen (P,G) advantages."""
    _finite(log_ratios,advantages)
    if log_ratios.ndim!=3 or advantages.shape!=log_ratios.shape[:2] or not 0<clip<1 or (log_ratios.detach().abs()>40).any():raise ValueError("invalid ratios/advantages")
    r=log_ratios.exp();a=advantages.detach()[:,:,None]
    return -torch.minimum(r*a,r.clamp(1-clip,1+clip)*a).mean()

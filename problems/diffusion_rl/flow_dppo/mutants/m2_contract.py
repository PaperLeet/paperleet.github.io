# Block equality at the divergence boundary
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

def policy_kl(current: Tensor, old: Tensor, std: Tensor) -> Tensor:
    """Exact equal-covariance Gaussian KL for (B,T,D) means and (T,) scales."""
    _finite(current,old,std)
    if current.ndim!=3 or old.shape!=current.shape or std.shape!=current.shape[1:2] or (std<=0).any():raise ValueError("invalid Gaussian shapes/scales")
    return ((current-old.detach()).square()/(2*std.detach()[None,:,None].square())).sum(-1)


def flow_dppo_loss(log_ratios: Tensor, advantages: Tensor, old_kl: Tensor, reference_kl: Tensor, threshold: float, beta: float) -> Tensor:
    """Directional divergence gating on (B,T) transitions; corrective updates always remain active."""
    _finite(log_ratios,advantages,old_kl,reference_kl)
    if log_ratios.ndim!=2 or old_kl.shape!=log_ratios.shape or reference_kl.shape!=log_ratios.shape or advantages.shape!=log_ratios.shape[:1] or not math.isfinite(threshold) or threshold<0 or not math.isfinite(beta) or beta<0 or (old_kl<0).any() or (reference_kl<0).any():raise ValueError("invalid divergence objective")
    adv=advantages.detach()[:,None];lr=log_ratios.detach()
    outward=((adv>0)&(lr>0))|((adv<0)&(lr<0))
    blocked=(old_kl.detach()>=threshold)&outward
    safe=torch.where(blocked,torch.zeros_like(log_ratios),log_ratios)
    if (safe.detach().abs()>40).any():raise ValueError("unmasked log ratios exceed domain")
    term=torch.where(blocked,torch.zeros_like(log_ratios),safe.exp()*adv)
    return (-term+beta*reference_kl).mean()

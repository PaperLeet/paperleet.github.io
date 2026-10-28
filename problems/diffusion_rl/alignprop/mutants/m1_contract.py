# Detach every suffix transition
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

def reward_rollout(initial: Tensor, alphas: Tensor, sigmas: Tensor, denoiser, backprop_from: int) -> Tensor:
    """Execute noisy-to-clean DDIM amplitudes; differentiate transitions from backprop_from onward."""
    _finite(initial,alphas,sigmas)
    if initial.ndim!=2 or alphas.ndim!=1 or sigmas.shape!=alphas.shape or len(alphas)<2 or (alphas<=0).any() or (sigmas<0).any():raise ValueError("invalid rollout/schedule")
    steps=len(alphas)-1
    if type(backprop_from) is not int or not 0<=backprop_from<steps:raise ValueError("invalid gradient boundary")
    a=alphas.detach();s=sigmas.detach();x=initial.detach()
    def step(state,i):
        epsilon=denoiser(state,i)
        if epsilon.shape!=state.shape:raise ValueError("denoiser changed shape")
        clean=(state-s[i]*epsilon)/a[i]
        return a[i+1]*clean+s[i+1]*epsilon
    for i in range(steps):
        if i<backprop_from:
            with torch.no_grad():x=step(x,i)
        else:
            x=x.detach()
            x=step(x,i)
    return x

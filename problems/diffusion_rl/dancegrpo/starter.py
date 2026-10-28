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
    'Replicate frozen (P,D) initial noise to (P,G,D), independently across prompts.'
    raise NotImplementedError


def group_advantages(rewards: Tensor, eps: float=1e-8) -> Tensor:
    '(P,G,K) rewards; sum independently population-normalized channel advantages.'
    raise NotImplementedError


def dance_loss(log_ratios: Tensor, advantages: Tensor, clip: float) -> Tensor:
    'Mean clipped policy loss for (P,G,T) ratios and frozen (P,G) advantages.'
    raise NotImplementedError

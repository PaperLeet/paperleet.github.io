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
    '(B,T,D) means/actions and positive (T,) scales; sum log densities over D.'
    raise NotImplementedError


def ddpo_loss(current: Tensor, old: Tensor, advantages: Tensor, clip: float) -> Tensor:
    '(B,T) transition log probabilities and (B,) frozen terminal advantages.'
    raise NotImplementedError

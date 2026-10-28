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
    'Zero-based completed-update index; sliding window saturates at the final valid start.'
    raise NotImplementedError


def mixed_transition(state: Tensor, velocity: Tensor, time: float, next_time: float, noise_level: float, stochastic: bool) -> tuple[Tensor,Tensor]:
    'Return Gaussian mean/std for SDE steps or deterministic ODE mean with zero std.'
    raise NotImplementedError


def window_loss(log_ratios: Tensor, advantages: Tensor, start: int, width: int, clip: float) -> Tensor:
    'Average clipped policy terms over the selected (B,width) window only.'
    raise NotImplementedError

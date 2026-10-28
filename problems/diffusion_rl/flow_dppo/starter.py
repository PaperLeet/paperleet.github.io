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
    'Exact equal-covariance Gaussian KL for (B,T,D) means and (T,) scales.'
    raise NotImplementedError


def flow_dppo_loss(log_ratios: Tensor, advantages: Tensor, old_kl: Tensor, reference_kl: Tensor, threshold: float, beta: float) -> Tensor:
    'Directional divergence gating on (B,T) transitions; corrective updates always remain active.'
    raise NotImplementedError

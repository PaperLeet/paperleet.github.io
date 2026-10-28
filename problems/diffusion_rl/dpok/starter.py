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

def gaussian_kl(current: Tensor, reference: Tensor, std: Tensor) -> Tensor:
    'Equal (B,T,D) means and shared positive (T,) std; return exact (B,T) KL.'
    raise NotImplementedError


def dpok_surrogate(logp: Tensor, rewards: Tensor, kl: Tensor, alpha: float, beta: float) -> Tensor:
    '(B,T) differentiable logp/KL and (B,) terminal rewards; sum T, mean B.'
    raise NotImplementedError

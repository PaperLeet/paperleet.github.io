import torch
from torch import Tensor


def group_norm(x: Tensor, num_groups: int, weight: Tensor, bias: Tensor,
               eps: float = 1e-5) -> Tensor:
    """Return [N,C,*spatial], using per-sample groups and per-channel affine.

    Preserve dtype, device and gradients. Invalid group counts raise ValueError.
    """
    raise NotImplementedError

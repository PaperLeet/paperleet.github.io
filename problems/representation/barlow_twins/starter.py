"""Single-batch Barlow Twins with the official affine-free BN convention."""
import math
import torch
from torch import Tensor


def cross_correlation(z1: Tensor, z2: Tensor, eps: float = 1e-5) -> Tensor:
    """Return [D,D] cross-correlation after featurewise batch standardization."""
    _check_inputs(z1, z2, eps)
    # Standardize each feature over the batch using population variance + eps.
    # Return normalized_z1.T @ normalized_z2 / B; keep gradients in both views.
    raise NotImplementedError


def barlow_twins_loss(z1: Tensor, z2: Tensor, lambd: float = .0051,
                     eps: float = 1e-5) -> Tensor:
    """Sum diagonal identity errors and lambd times all off-diagonal squares."""
    _check_scalar(lambd, 'lambd', positive=False)
    # Call cross_correlation; sum diagonal identity errors and all off-diagonal squares.
    # Return a scalar tensor; lambd=0 removes only the redundancy penalty.
    raise NotImplementedError


# Provided validation; implement the two numerical functions above.
def _check_scalar(value, name, positive=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + ' must be a finite Python number')
    if value <= 0 if positive else value < 0:
        raise ValueError(name + (' must be positive' if positive else ' must be nonnegative'))


def _check_inputs(z1, z2, eps):
    if z1.ndim != 2 or z2.shape != z1.shape or z1.shape[0] < 2 or z1.shape[1] < 1:
        raise ValueError('views must have the same [B,D] shape, B>=2 and D>=1')
    if z1.dtype not in (torch.float32, torch.float64) or z2.dtype != z1.dtype or z1.device != z2.device:
        raise ValueError('views must share float32/float64 dtype and device')
    if not bool(torch.isfinite(z1).all()) or not bool(torch.isfinite(z2).all()):
        raise ValueError('views must contain finite values')
    _check_scalar(eps, 'eps')

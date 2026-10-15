"""VICReg's three losses, using the official single-process normalization."""
import math
import torch
from torch import Tensor


def _check_pair(x, y, eps):
    """Provided input validation.

    Returns:
        Provided validator: return Python None when finite matching float32/float64 views [B,D]
        satisfy B>=2, D>=1 and epsilon is finite positive. Raise ValueError when a checked
        condition fails; do not mutate inputs or return transformed tensors.
    """
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1 or x.dtype not in (torch.float32, torch.float64):
        raise ValueError("x must be float32/float64 (B,D), B>=2, D>=1")
    if y.shape != x.shape or y.dtype != x.dtype or y.device != x.device:
        raise ValueError("x and y must have the same shape, dtype and device")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("views must be finite")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")


def _check_weights(sim_coeff, std_coeff, cov_coeff):
    """Provided input validation.

    Returns:
        Provided validator: return Python None when all three coefficients are finite and
        nonnegative. Raise ValueError when a checked condition fails; do not mutate inputs or
        return transformed tensors.
    """
    if any(not math.isfinite(v) or v < 0 for v in (sim_coeff, std_coeff, cov_coeff)):
        raise ValueError("coefficients must be finite and nonnegative")


def vicreg_terms(x: Tensor, y: Tensor, eps: float = 1e-4) -> tuple[Tensor, Tensor, Tensor]:
    """Return scalar invariance, variance and covariance terms, in that order.

    Returns:
        Return (invariance, variance, covariance), exactly three scalar floating Tensors with
        shape () and the views' dtype/device. For views [B,D], invariance averages B*D squared
        differences; variance averages feature penalties and both views; covariance sums both
        views' off-diagonal squared entries and divides by D. Preserve gradients to both views,
        including sample statistics with denominator B-1; do not detach or mutate inputs.
    """
    _check_pair(x, y, eps)
    raise NotImplementedError("Implement vicreg_terms")


def vicreg_loss(x: Tensor, y: Tensor, sim_coeff: float = 25., std_coeff: float = 25.,
                cov_coeff: float = 1., eps: float = 1e-4) -> Tensor:
    """Keep gradients in both views and combine the three scalar terms.

    Returns:
        Return one scalar floating Tensor, shape (), equal to the weighted sum of the three
        vicreg_terms outputs in their declared order. Preserve the common input dtype/device and
        gradients to both x and y. Do not return the components, a Python float, a detached
        value or an in-place update.
    """
    _check_weights(sim_coeff, std_coeff, cov_coeff)
    raise NotImplementedError("Implement vicreg_loss")

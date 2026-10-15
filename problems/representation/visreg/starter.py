"""VISReg Algorithm 1 with explicit projections and the epsilon from Eq. (2)."""
import math
import torch
from torch import Tensor


def _features(x, ndim):
    """Provided input validation.

    Returns:
        Provided validator: return Python None when x is a finite, nonempty float32/float64
        Tensor of the requested rank. Raise ValueError when a checked condition fails; do not
        mutate inputs or return transformed tensors.
    """
    if not isinstance(x, Tensor) or x.ndim != ndim or min(x.shape) < 1 or x.dtype not in (torch.float32, torch.float64) or not torch.isfinite(x).all():
        raise ValueError('expected finite, nonempty float32/float64 features')


def _check_terms(z, directions, eps):
    """Provided input validation.

    Returns:
        Provided validator: return Python None when features z [N,D], N>=2, and matching finite
        directions [D,K] have nonzero columns and epsilon is finite positive. Raise ValueError
        when a checked condition fails; do not mutate inputs or return transformed tensors.
    """
    _features(z, 2); _features(directions, 2)
    if z.shape[0] < 2 or directions.shape[0] != z.shape[1] or directions.dtype != z.dtype or directions.device != z.device:
        raise ValueError('need N>=2, directions [D,K] and matching dtype/device')
    if (directions.square().sum(0) == 0).any() or not math.isfinite(eps) or eps <= 0:
        raise ValueError('nonzero directions and positive finite epsilon required')


def visreg_terms(z: Tensor, directions: Tensor, eps: float = 1e-4) -> tuple[Tensor, Tensor, Tensor]:
    """Return (scale, shape, center), all scalars; supplied directions are fixed.

    Returns:
        Return (scale, shape, center), exactly three scalar floating Tensors with shape () and
        z's dtype/device. For z [N,D] and K directions, scale and center average D features;
        shape averages N*K projected quantile errors. Keep z gradients, but freeze supplied
        directions and freeze standard deviation only in the shape denominator. The scale term
        retains its standard-deviation gradient. No input mutation.
    """
    raise NotImplementedError("Implement visreg_terms")


def view_invariance(views: Tensor, num_global: int = 2) -> Tensor:
    """Mean squared error of all [V,N,D] views from the first G views' mean.

    Returns:
        Return one scalar floating Tensor, shape (), with views' dtype/device: squared error
        from the first G=num_global views' mean, averaged over all V*N*D entries of views
        [V,N,D]. Gradients flow through every view and through that global-view mean; it is not
        a detached teacher. Leave inputs unchanged.
    """
    raise NotImplementedError("Implement view_invariance")

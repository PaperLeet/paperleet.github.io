"""LeJEPA SIGReg with supplied fixed directions and positive quadrature nodes."""
import torch
from torch import Tensor


def _check_projected(projected, t):
    """Provided input validation.

    Returns:
        Provided validator: return Python None when projected [N,M] is finite nonempty
        float32/float64 and matching nodes t [K], K>=2, start at zero and strictly increase.
        Raise ValueError when a checked condition fails; do not mutate inputs or return
        transformed tensors.
    """
    if projected.ndim != 2 or min(projected.shape) < 1 or projected.dtype not in (torch.float32, torch.float64):
        raise ValueError("projected must be nonempty float32/float64 (N,M)")
    if t.ndim != 1 or t.numel() < 2 or t.dtype != projected.dtype or t.device != projected.device:
        raise ValueError("t must be a matching vector with at least two nodes")
    if not torch.isfinite(projected).all() or not torch.isfinite(t).all():
        raise ValueError("inputs must be finite")
    if t[0] != 0 or not (t[1:] > t[:-1]).all():
        raise ValueError("t must start at zero and increase strictly")


def _check_features(features, directions, t):
    """Provided input validation.

    Returns:
        Provided validator: return Python None when features [N,D] and nodes pass validation and
        finite matching directions [D,M] have unit-length columns. Raise ValueError when a
        checked condition fails; do not mutate inputs or return transformed tensors.
    """
    _check_projected(features, t)
    if directions.ndim != 2 or directions.shape[0] != features.shape[1] or directions.shape[1] < 1:
        raise ValueError("directions must have shape (D,M), M>=1")
    if directions.dtype != features.dtype or directions.device != features.device or not torch.isfinite(directions).all():
        raise ValueError("directions must match feature dtype/device and be finite")
    tolerance = 100 * torch.finfo(features.dtype).eps
    if not torch.allclose(directions.norm(dim=0), torch.ones_like(directions[0]), atol=tolerance, rtol=tolerance):
        raise ValueError("each direction must have unit Euclidean norm")


def epps_pulley(projected: Tensor, t: Tensor) -> Tensor:
    """Return one N-scaled Epps-Pulley statistic per projection; t is fixed.

    Returns:
        Return one floating Tensor [M], one N-scaled Epps-Pulley statistic for each of M
        projection columns in projected[N,M]. Integrate out the quadrature-node axis; do not
        average the M slices here. Preserve dtype/device and gradients to projected samples,
        detach fixed nodes t, and leave inputs unchanged.
    """
    _check_projected(projected, t)
    raise NotImplementedError("Implement epps_pulley")


def sigreg_loss(features: Tensor, directions: Tensor, t: Tensor) -> Tensor:
    """Mean over supplied unit directions; no centering, whitening or feature detach.

    Returns:
        Return one scalar floating Tensor, shape (), equal to the mean of the M slice statistics
        for features [N,D] and directions [D,M]. Preserve feature dtype/device and gradients;
        both directions and quadrature nodes t are detached constants. Do not further divide by
        N, detach features or mutate inputs.
    """
    _check_features(features, directions, t)
    raise NotImplementedError("Implement sigreg_loss")

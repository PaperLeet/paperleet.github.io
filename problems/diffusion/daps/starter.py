"""DAPS Eq. (7), specialized to a supplied linear measurement operator."""
import math
import torch
from torch import Tensor


def _positive(value, name, allow_zero=False):
    """Provided helper.

    Returns:
        Provided: return Python None when the scalar is valid, not a boolean or a validated
        tensor. Invalid inputs raise ValueError. Validation only: leave inputs and their
        gradient settings unchanged. No returned tensor shape or dtype/device applies.
    """
    if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name} has an invalid value")


def _matching(x, *others):
    """Provided helper.

    Returns:
        Provided: return Python None when floating tensors have finite values and matching
        shape/dtype/device, not a boolean or a validated tensor. Invalid inputs raise
        ValueError. Validation only: leave inputs and their gradient settings unchanged. No
        returned tensor shape or dtype/device applies.
    """
    if not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError("finite floating tensors required")
    for other in others:
        if other.shape != x.shape or other.dtype != x.dtype or other.device != x.device or not torch.isfinite(other).all():
            raise ValueError("tensor shape/dtype/device must match")


def posterior_score(x: Tensor, x0_hat: Tensor, operator: Tensor, measurement: Tensor,
                    r: float, beta: float) -> Tensor:
    """Score of N(x0_hat, r^2 I) times N(measurement; A x, beta^2 I).

    Returns:
        Return per-sample conditional scores as a Tensor [B, D], matching x shape and
        dtype/device; B counts samples and D unknown coordinates. Measurement width M is not
        an output axis. Do not reduce across samples/coordinates or divide by B or D. Keep
        differentiable tensor operations and leave inputs unchanged.
    """
    _positive(r, "r"); _positive(beta, "beta")
    _matching(x, x0_hat)
    if x.ndim != 2 or operator.ndim != 2 or operator.shape[1] != x.shape[1]:
        raise ValueError("x is (B,D); operator is (M,D)")
    if measurement.shape != (x.shape[0], operator.shape[0]):
        raise ValueError("measurement must be (B,M)")
    for value in (operator, measurement):
        if value.dtype != x.dtype or value.device != x.device or not torch.isfinite(value).all():
            raise ValueError("finite matching dtype/device required")
    raise NotImplementedError("Implement the paper core")


def langevin_step(x: Tensor, score: Tensor, eta: float, noise: Tensor) -> Tensor:
    """Euler Langevin step; caller supplies fresh standard-normal noise explicitly.

    Returns:
        Return the updated sample Tensor with exactly x shape and dtype/device ([B,D] in
        this exercise, for B samples and D coordinates). Return only the new sample, not the
        score or noise. eta=0 gives x numerically. Do not mutate inputs or explicitly detach
        the output.
    """
    _positive(eta, "eta", allow_zero=True)
    _matching(x, score, noise)
    raise NotImplementedError("Implement the paper core")


def renoise(x0: Tensor, sigma_next: float, noise: Tensor) -> Tensor:
    """Decoupled transition N(x0, sigma_next^2 I), with fresh independent noise.

    Returns:
        Return one next-level noisy sample Tensor with exactly x0 shape and dtype/device
        ([B,D] here). sigma_next=0 gives x0 numerically. Keep differentiable tensor
        operations and do not mutate x0 or the supplied noise.
    """
    _positive(sigma_next, "sigma_next", allow_zero=True)
    _matching(x0, noise)
    raise NotImplementedError("Implement the paper core")

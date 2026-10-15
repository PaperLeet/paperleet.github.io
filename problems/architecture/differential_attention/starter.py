import math
import torch
from torch import Tensor


def differential_lambda(q1: Tensor, k1: Tensor, q2: Tensor, k2: Tensor,
                        lambda_init: float = 0.2) -> Tensor:
    """The paper's scalar exponential-difference parameterization.

    Returns:
        Return one zero-dimensional floating Tensor (`shape == ()`) containing the scalar
        coefficient lambda, not a Python float or one-element vector. It is not restricted
        to [0,1]. Preserve the parameter vectors' dtype/device and gradients to all four
        vectors, without changing inputs.
    """
    if q1.ndim != 1 or q1.numel() == 0 or any(x.shape != q1.shape for x in (k1,q2,k2)):
        raise ValueError("lambda parameters must have equal nonempty vector shapes")
    if not math.isfinite(lambda_init):
        raise ValueError("lambda_init must be finite")
    raise NotImplementedError("Compute the scalar exponential difference")


def differential_attention(q1: Tensor, k1: Tensor, q2: Tensor, k2: Tensor,
                           value: Tensor, coefficient, causal: bool = True) -> Tensor:
    """Projected single/multiple heads, before head normalization and output projection.

    Returns:
        Return one floating Tensor `[B,H,T,2D]`, with batch size B, head count H, sequence
        length T and Q/K width D; its shape matches value. This is the differential value
        aggregate before head normalization/output projection, not a tuple containing
        attention weights. Preserve input dtype/device and gradients through both Q/K
        branches, value and a Tensor coefficient. Do not modify inputs.
    """
    if q1.ndim != 4 or q1.numel() == 0 or any(x.shape != q1.shape for x in (k1,q2,k2)):
        raise ValueError("Q/K must have equal nonempty [B,H,T,D] shapes")
    if value.shape != (*q1.shape[:-1], 2*q1.shape[-1]):
        raise ValueError("V must have shape [B,H,T,2D]")
    if isinstance(coefficient, Tensor) and coefficient.ndim != 0:
        raise ValueError("coefficient must be scalar")
    if not math.isfinite(float(coefficient.detach() if isinstance(coefficient, Tensor) else coefficient)):
        raise ValueError("coefficient must be finite")
    raise NotImplementedError("Mask and normalize both branches before subtracting")

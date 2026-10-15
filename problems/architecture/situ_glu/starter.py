import math
import torch
from torch import Tensor


def softcap(x: Tensor, beta: float) -> Tensor:
    """Smoothly cap a floating tensor using a finite positive scalar beta.

    Returns:
        Return one floating Tensor with exactly x's shape, dtype and device, including
        scalar or empty tensors. Each value is smoothly capped to the range [-beta,beta]
        (endpoints may be reached numerically). Preserve gradients into x; do not mutate x
        or return the cap parameter alongside the result.
    """
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    raise NotImplementedError("Implement softcap")


def situ_glu(gate: Tensor, up: Tensor, beta1: float = 4.0,
             beta2: float = 25.0) -> Tensor:
    """Apply SiTU-GLU to two already projected, equally shaped branches.

    Returns:
        Return one floating Tensor matching the shared shape/dtype/device of gate and up,
        retaining arbitrary leading dimensions and empty axes. It is the elementwise gated
        activation, bounded in magnitude by beta1*beta2; no feature dimension is reduced.
        Keep gradients to both branches and leave inputs unchanged.
    """
    raise NotImplementedError("Implement situ_glu")

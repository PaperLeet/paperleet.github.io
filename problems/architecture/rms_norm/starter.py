import torch
from torch import Tensor


def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
    """Required: x [..., D], weight [D], finite float32/float64, D >= 1.
    Normalize each last-axis vector by sqrt(mean(x**2) + eps), then scale.
    No centering or bias. eps > 0. Preserve shape, dtype, device and gradients;
    do not mutate inputs. Use tensor operations, not a normalization API.

    Returns:
        Return one floating Tensor with x's full `[...,D]` shape, where D is feature width
        and all leading batch/token axes are preserved. Return scaled normalized features,
        not the RMS statistic. Preserve x's dtype/device and gradients to x and weight
        without changing inputs.
    """
    raise NotImplementedError()


def partial_rms_norm(x: Tensor, weight: Tensor, k: int, eps: float = 1e-5) -> Tensor:
    """Required: same contract, but estimate mean square from x[..., :k].
    Return all D normalized features. The prefix statistic remains differentiable.
    Reject non-int/bool k and k outside [1, D] with ValueError.
    k is an explicit feature count, so percentage rounding is outside this task.

    Returns:
        Return one floating Tensor `[...,D]`, preserving all D features and all leading
        dimensions even though only the first k features determine the normalization scale.
        D is feature width. Preserve dtype/device and gradients to x and weight, including
        the dependence of every output on the prefix statistic. Do not detach the statistic
        or modify inputs.
    """
    raise NotImplementedError()

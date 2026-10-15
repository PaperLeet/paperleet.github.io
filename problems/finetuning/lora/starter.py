import torch
from torch import Tensor

def lora_delta(a: Tensor, b: Tensor, alpha: float=1.0) -> Tensor:
    """A [r,in], B [out,r] -> delta W [out,in], scaled by alpha/r.

    Returns:
        Return one floating Tensor `[out_features,in_features]`, the scaled low-rank weight
        update, from a `[r,in_features]` and b `[out_features,r]`. Preserve their shared
        dtype/device and gradients to both adapters. Return the update matrix only, without
        merging it into or modifying a base weight.
    """
    raise NotImplementedError('Implement lora_delta')

def lora_linear(x: Tensor, weight: Tensor, a: Tensor, b: Tensor, alpha: float=1.0, bias: Tensor | None=None) -> Tensor:
    """Frozen base and bias; gradients still flow to x and both adapters.

    Returns:
        Return one floating Tensor `[...,out_features]` from x `[...,in_features]`,
        preserving every leading batch/token dimension. It contains the base linear output
        plus adapter contribution, with optional bias included once. Preserve x's
        dtype/device and gradients to x, a and b; weight and optional bias are frozen even
        if they require gradients. Do not mutate any input or return updated parameters.
    """
    raise NotImplementedError('Implement lora_linear')

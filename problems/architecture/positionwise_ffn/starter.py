import torch
from torch import Tensor


def positionwise_ffn(x: Tensor, w1: Tensor, b1: Tensor,
                     w2: Tensor, b2: Tensor) -> Tensor:
    """x [..., D], w1 [H, D], b1 [H], w2 [D, H], b2 [D]; no dropout.

    Returns:
        Return one floating Tensor `[...,D]`, matching x's complete shape: D is the model
        feature width and leading batch/token axes are unchanged. Return the second linear
        layer's output, not hidden activations of width H. Preserve dtype/device and
        gradients to x, w1, b1, w2 and b2. Each token stays independent and inputs are not
        modified.
    """
    raise NotImplementedError("Implement positionwise_ffn")

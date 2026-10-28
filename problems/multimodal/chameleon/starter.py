import math
import torch
from torch import Tensor
import torch.nn.functional as F


def _floating(*values):
    first = values[0]
    for value in values:
        if not value.is_floating_point() or value.dtype != first.dtype or value.device != first.device or not torch.isfinite(value).all():
            raise ValueError("floating inputs must be finite and share dtype/device")

def qk_layer_norm(x: Tensor, weight: Tensor, bias: Tensor, eps: float) -> Tensor:
    """x is [T,H,D]; weight/bias are [H,D]. Compute mean and population variance only over the last dimension D for each token/head; add positive eps inside the square root. Affine parameters are independent per head. Preserve all input gradients. Apply this kernel separately to queries and keys; RoPE and full multimodal generation are excluded."""
    raise NotImplementedError("Implement qk_layer_norm")

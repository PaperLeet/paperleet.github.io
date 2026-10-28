import math
import torch
from torch import Tensor
import torch.nn.functional as F


def _floating(*values):
    first = values[0]
    for value in values:
        if not value.is_floating_point() or value.dtype != first.dtype or value.device != first.device or not torch.isfinite(value).all():
            raise ValueError("floating inputs must be finite and share dtype/device")

def align_loss(images: Tensor, texts: Tensor, temperature: Tensor) -> Tensor:
    """images and texts are nonempty floating [B,D] tensors; temperature is a positive scalar tensor. Matching indices are positive pairs. Normalize rows with norm floor 1e-12, keeping zero rows zero. Average over the batch in each retrieval direction, then average the two directions. Preserve all three gradients."""
    raise NotImplementedError("Implement align_loss")

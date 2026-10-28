import math
import torch
from torch import Tensor
import torch.nn.functional as F


def _floating(*values):
    first = values[0]
    for value in values:
        if not value.is_floating_point() or value.dtype != first.dtype or value.device != first.device or not torch.isfinite(value).all():
            raise ValueError("floating inputs must be finite and share dtype/device")

def max_query_similarity(queries: Tensor, texts: Tensor, temperature: Tensor) -> Tensor:
    """queries is [B,Q,D], texts is [N,D], and temperature is a positive scalar. Normalize along D and return [B,N] by maximizing independently over Q for every image-text pair. At ties, use torch.max first-index gradients. This kernel does not implement the full Q-Former."""
    raise NotImplementedError("Implement max_query_similarity")

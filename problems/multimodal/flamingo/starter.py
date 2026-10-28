import math
import torch
from torch import Tensor
import torch.nn.functional as F


def _floating(*values):
    first = values[0]
    for value in values:
        if not value.is_floating_point() or value.dtype != first.dtype or value.device != first.device or not torch.isfinite(value).all():
            raise ValueError("floating inputs must be finite and share dtype/device")

def gated_media_attention(x: Tensor, keys: Tensor, values: Tensor, text_media: Tensor, visual_media: Tensor, gate: Tensor) -> Tensor:
    """x is [T,D], keys/values are [N,D], and text_media/visual_media are int64 [T]/[N]. Media IDs start at 1; each text ID names its immediately preceding medium, with 0 for no medium. Only matching IDs are visible; rows without visible media return x exactly. gate is scalar and scales the residual through tanh. Use one head and identity projections, without Perceiver or FFN."""
    raise NotImplementedError("Implement gated_media_attention")

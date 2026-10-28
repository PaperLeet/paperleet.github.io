import math
import torch
from torch import Tensor
import torch.nn.functional as F


def _floating(*values):
    first = values[0]
    for value in values:
        if not value.is_floating_point() or value.dtype != first.dtype or value.device != first.device or not torch.isfinite(value).all():
            raise ValueError("floating inputs must be finite and share dtype/device")

def variable_visual_positions(visual: Tensor, image_ids: Tensor, deltas: Tensor) -> Tensor:
    """visual is boolean [L]; image_ids is int64 [L], using an image ID in [0,M) for visual tokens and -1 for text. deltas is floating [M] with 0<delta<=1. All tokens of an image share its delta. Start at p_-1=-1 and add 1 for the current text token or its image delta for the current visual token. Return [L] with deltas dtype/device. Empty sequences and text-only M=0 are valid; preserve delta gradients."""
    raise NotImplementedError("Implement variable_visual_positions")

import math
import torch
from torch import Tensor
import torch.nn.functional as F


def _floating(*values):
    first = values[0]
    for value in values:
        if not value.is_floating_point() or value.dtype != first.dtype or value.device != first.device or not torch.isfinite(value).all():
            raise ValueError("floating inputs must be finite and share dtype/device")

def pack_visual_tokens(text: Tensor, visual: Tensor, labels: Tensor, image_index: int) -> tuple[Tensor, Tensor]:
    """One sequence: text is [T,D] with one image placeholder, visual is projected [P,D], and labels is int64 [T]. Replace the token at image_index with all P visual tokens; fill their labels with -100 and retain all other labels. Return length T+P-1 embeddings and labels. Gradients reach retained text and visual features; the removed placeholder receives zero gradient."""
    raise NotImplementedError("Implement pack_visual_tokens")

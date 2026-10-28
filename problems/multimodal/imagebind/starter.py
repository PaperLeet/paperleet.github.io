import math
import torch
from torch import Tensor
import torch.nn.functional as F


def _floating(*values):
    first = values[0]
    for value in values:
        if not value.is_floating_point() or value.dtype != first.dtype or value.device != first.device or not torch.isfinite(value).all():
            raise ValueError("floating inputs must be finite and share dtype/device")

def image_anchor_loss(images: Tensor, modalities: Tensor, temperature: Tensor) -> Tensor:
    """images is [B,D] and modalities [M,B,D], with positive dimensions. Every modality is paired with the same image batch; do not add direct losses between nonimage modalities. Normalize features, compute bidirectional batch-softmax losses per image-modality pair, and average over M. temperature is a positive scalar. This teaching kernel permits image gradients; freezing the image encoder is the caller’s responsibility."""
    raise NotImplementedError("Implement image_anchor_loss")

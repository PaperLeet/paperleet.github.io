# BUG: Drops text-to-image contrast
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
    _floating(images, texts, temperature)
    if images.ndim != 2 or images.numel() == 0 or texts.shape != images.shape or temperature.ndim != 0 or temperature.item() <= 0:
        raise ValueError("expected matching nonempty (B,D) features and positive scalar temperature")
    x = F.normalize(images, dim=-1, eps=1e-12)
    y = F.normalize(texts, dim=-1, eps=1e-12)
    scores = x @ y.T / temperature
    return -F.log_softmax(scores, dim=1).diag().mean()

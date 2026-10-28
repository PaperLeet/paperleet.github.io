import math
import torch
from torch import Tensor
import torch.nn.functional as F


def _floating(*values):
    first = values[0]
    for value in values:
        if not value.is_floating_point() or value.dtype != first.dtype or value.device != first.device or not torch.isfinite(value).all():
            raise ValueError("floating inputs must be finite and share dtype/device")

def masked_patch_distillation(student: Tensor, teacher_prob: Tensor, mask: Tensor, temperature: float) -> Tensor:
    """student and teacher_prob have shape [B,P,K]. Teacher probabilities are nonnegative and sum to one over K; teacher projection and normalization are already done. mask is boolean [B,P] with at least one selected patch. temperature is a positive Python float. Average cross-entropy only over masked patches, stop teacher gradients, and preserve student gradients."""
    raise NotImplementedError("Implement masked_patch_distillation")

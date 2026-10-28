import math
import torch
from torch import Tensor
import torch.nn.functional as F


def _floating(*values):
    first = values[0]
    for value in values:
        if not value.is_floating_point() or value.dtype != first.dtype or value.device != first.device or not torch.isfinite(value).all():
            raise ValueError("floating inputs must be finite and share dtype/device")

def momentum_target_loss(logits: Tensor, teacher_logits: Tensor, positives: Tensor, alpha: float) -> Tensor:
    """logits and teacher_logits are [B,N], where N may include queued negatives. positives is int64 [B] of positive columns. alpha is in [0,1]; targets mix alpha*teacher_softmax with (1-alpha)*one_hot. Stop teacher gradients and average over B, not B*N. Inputs are already temperature-scaled."""
    raise NotImplementedError("Implement momentum_target_loss")

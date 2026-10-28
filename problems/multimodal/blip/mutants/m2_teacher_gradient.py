# BUG: Updates momentum teacher through targets
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
    _floating(logits, teacher_logits)
    if logits.ndim!=2 or logits.numel()==0 or logits.shape!=teacher_logits.shape or positives.shape!=(logits.shape[0],) or positives.dtype!=torch.long or not 0<=alpha<=1:
        raise ValueError("expected (B,N) logits, (B,) long positives and alpha in [0,1]")
    if (positives<0).any() or (positives>=logits.shape[1]).any(): raise ValueError("positive index out of range")
    hard=(torch.arange(logits.shape[1],device=logits.device)[None,:]==positives[:,None]).to(logits.dtype)
    targets=alpha*torch.softmax(teacher_logits,-1)+(1-alpha)*hard
    return -(targets*F.log_softmax(logits,-1)).sum(-1).mean()

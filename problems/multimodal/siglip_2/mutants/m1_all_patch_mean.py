# BUG: Divides by all patches
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
    _floating(student, teacher_prob)
    if student.ndim != 3 or student.numel()==0 or teacher_prob.shape!=student.shape or mask.shape!=student.shape[:2] or mask.dtype!=torch.bool or not mask.any() or temperature<=0:
        raise ValueError("expected (B,P,K), nonempty boolean mask, and positive temperature")
    if (teacher_prob<0).any() or not torch.allclose(teacher_prob.sum(-1),torch.ones_like(teacher_prob.sum(-1)),atol=1e-6,rtol=1e-6):
        raise ValueError("teacher targets must be probabilities")
    per_patch=-(teacher_prob.detach()*F.log_softmax(student/temperature,dim=-1)).sum(-1)
    return (per_patch*mask.to(student.dtype)).sum()/mask.numel()

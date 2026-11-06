# BUG: Ignore student temperature.
"""The same-view masked-patch branch of iBOT, with stopped centered teacher targets."""
import math
import torch
from torch import Tensor


def teacher_patch_targets(teacher_logits: Tensor, center: Tensor,
                          teacher_temp: float = .04) -> Tensor:
    """Return detached [B,N,K] soft targets; center is a fixed [K] vector."""
    _teacher(teacher_logits, center, teacher_temp)
    return ((teacher_logits - center) / teacher_temp).softmax(-1).detach()


def ibot_patch_loss(student_logits: Tensor, teacher_logits: Tensor, mask: Tensor,
                    center: Tensor, student_temp: float = .1,
                    teacher_temp: float = .04) -> Tensor:
    """Average masked CE within each image, then average all B images, including empty ones."""
    _check(student_logits, teacher_logits, mask, center, student_temp, teacher_temp)
    targets = teacher_patch_targets(teacher_logits, center, teacher_temp)
    patch_loss = -(targets * student_logits.log_softmax(-1)).sum(-1)
    masked = patch_loss * mask.to(student_logits.dtype)
    return (masked.sum(-1) / mask.sum(-1).clamp(min=1)).mean()


# Provided input checks; full iBOT also has a class-token objective and teacher/center EMA.
def _features(x):
    if x.ndim != 3 or min(x.shape) < 1 or x.dtype not in (torch.float32, torch.float64):
        raise ValueError('expected nonempty float32/float64 [B,N,K]')
    if not bool(torch.isfinite(x).all()):
        raise ValueError('logits must be finite')


def _temperature(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('temperature must be a finite positive Python number')


def _teacher(teacher_logits, center, teacher_temp):
    _features(teacher_logits); _temperature(teacher_temp)
    if center.shape != (teacher_logits.shape[-1],) or center.dtype != teacher_logits.dtype or center.device != teacher_logits.device:
        raise ValueError('center must share teacher dtype/device and have shape [K]')
    if not bool(torch.isfinite(center).all()):
        raise ValueError('center must be finite')


def _check(student_logits, teacher_logits, mask, center, student_temp, teacher_temp):
    _features(student_logits); _teacher(teacher_logits, center, teacher_temp); _temperature(student_temp)
    if student_logits.shape != teacher_logits.shape or student_logits.dtype != teacher_logits.dtype or student_logits.device != teacher_logits.device:
        raise ValueError('student and teacher logits must share shape, dtype and device')
    if mask.shape != student_logits.shape[:2] or mask.dtype != torch.bool or mask.device != student_logits.device:
        raise ValueError('mask must be boolean [B,N] on the logits device')

import math
import torch
from torch import Tensor


def _float(x, ndim=None):
    if not isinstance(x, Tensor) or not x.is_floating_point() or x.numel() == 0 or (ndim is not None and x.ndim != ndim) or not torch.isfinite(x).all():
        raise ValueError("expected a nonempty finite floating tensor with the declared rank")


def _same(a, b):
    _float(b)
    if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
        raise ValueError("tensor shape, dtype and device must match")

def action_probabilities(logits: Tensor, action_tokens: Tensor, is_robot: Tensor) -> Tensor:
    """Constrain robot rows to action tokens, while preserving the full VLM vocabulary."""
    _float(logits,2)
    if action_tokens.shape!=logits.shape[1:] or is_robot.shape!=logits.shape[:1] or action_tokens.dtype!=torch.bool or is_robot.dtype!=torch.bool or action_tokens.device!=logits.device or is_robot.device!=logits.device or not action_tokens.any(): raise ValueError("invalid task or vocabulary mask")
    allowed=action_tokens.unsqueeze(0)|(~is_robot).unsqueeze(1)
    return torch.softmax(logits.masked_fill(~allowed,float('-inf')),dim=-1)

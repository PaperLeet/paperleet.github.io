# BUG: Omit reciprocal keep-probability scaling
import math
import torch


def inverted_dropout(x, p, mask=None, training=True):
    if x.dtype not in (torch.float32, torch.float64) or not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("floating input and finite drop probability in [0,1] required")
    if not training or p == 0:
        return x
    if p == 1:
        return x * 0.
    if mask is None or mask.shape != x.shape or mask.dtype != torch.bool or mask.device != x.device:
        raise ValueError("one explicit boolean keep mask per element required")
    return x * mask.to(x.dtype)

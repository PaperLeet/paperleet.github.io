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

def mixture_actions(expert: Tensor, learner: Tensor, draws: Tensor, beta: float) -> Tensor:
    'Choose expert iff draws[b]<beta; actions (B,A), draws (B,).'
    raise NotImplementedError("Implement mixture_actions according to the README contract")


def aggregate_dataset(old_states: Tensor, old_labels: Tensor, states: Tensor, labels: Tensor) -> tuple:
    'Append all expert-labeled visited states; return detached (states, labels).'
    raise NotImplementedError("Implement aggregate_dataset according to the README contract")

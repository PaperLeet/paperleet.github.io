from __future__ import annotations
import torch


def inverted_dropout(x: torch.Tensor, p: float, mask: torch.Tensor | None = None, training: bool = True) -> torch.Tensor:
    """
    Implement explicitly masked inverted dropout, distinguishing training, inference and both probability endpoints.

    x is a finite float32/float64 tensor of any shape, including scalars, empty dimensions and strided views. **p is the drop probability**, a finite real number in `[0,1]`; the paper also uses p for keep probability, so translate that notation.

    For training with `0<p<1`, mask must be a bool **keep** mask with exactly the same shape/device as x. No broadcasting or internal sampling: True retains and rescales, False drops. At p=0 or in inference, ignore mask and return equal input values. Inference remains identity even at p=1. Training at p=1 returns graph-connected zero with zero gradients and needs no mask. Preserve dtype/device/shape and input gradients without mutating x or mask; fresh storage is not promised. Invalid probabilities/dtypes/stochastic-training masks raise ValueError.
    """
    raise NotImplementedError

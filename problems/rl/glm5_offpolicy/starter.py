import math
import torch
from torch import Tensor


def masked_is_weights(logp: Tensor, rollout_logp: Tensor, mask: Tensor,
                      clip_low: float = .2, clip_high: float = .2) -> Tensor:
    """Return detached r=exp(logp-rollout_logp) only when 1-low<r<1+high. Inputs share shape
    [B,L], mask is binary, output has logp's shape/dtype. Both endpoints, off-policy
    outliers and padding receive zero, not clamping. Compare the computed ratio in its
    native dtype; safely replace rejected values (even detached exp overflow) with zero
    using selection, not 0*inf. Reject shape mismatches; require 0<clip_low<1 and finite
    clip_high>0.

    Returns:
        Return one detached floating Tensor weights, shape [B, L], matching logp
        dtype/device (B responses, L tokens). Retained positions contain
        exp(logp-rollout_logp) strictly inside (1-clip_low, 1+clip_high); padding, boundary
        and rejected positions are zero. Return the full token array, without reduction or
        input mutation.
    """
    raise NotImplementedError()


def offpolicy_loss(logp: Tensor, rollout_logp: Tensor, advantages: Tensor, mask: Tensor,
                   clip_low: float = .2, clip_high: float = .2) -> Tensor:
    """Return -sum(weights * stopgrad(advantages[:,None]) * logp)/sum(mask). advantages is [B].
    Freeze weights and rollout scores; gradient flows only through the direct current-logp
    factor. Normalize by all original valid tokens, including those dropped by the ratio
    rule. Empty mask: graph-safe 0. Ignore padding before arithmetic and do not mutate
    inputs.

    Returns:
        Return one scalar floating Tensor loss, shape (), normalized by the number of
        original valid tokens, including ratio-rejected tokens. Gradient flows only through
        the direct current-logp factor; weights, rollout scores and advantages are fixed. An
        empty mask returns a graph-connected zero so .backward() remains valid; never return
        .item() or a detached scalar.
    """
    raise NotImplementedError()

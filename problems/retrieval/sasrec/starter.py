"""SASRec causal/padding visibility and stable sampled next-item objective."""
import torch
from torch import Tensor
from torch.nn import functional as F


def causal_item_mask(item_ids: Tensor) -> Tensor:
    """[B,T] int64 IDs (0 = padding) -> [B,T,T] Boolean query/key visibility.

    Returns:
        Return one Boolean Tensor [B,T,T] on the item-ID device; its axes are batch, query
        position i, key position j. True means both IDs are nonpadding and j<=i, including the
        diagonal. Empty axes remain valid. Return visibility, not attention weights or an
        additive float mask; no gradients or input mutation.
    """
    _check_ids(item_ids)
    raise NotImplementedError('Implement causal_item_mask')


def sampled_next_item_loss(positive: Tensor, negative: Tensor, target_ids: Tensor) -> Tensor:
    """Mean softplus(-positive)+softplus(negative) over nonzero target positions.

    Returns:
        Return one scalar floating Tensor, shape (), with the logits' dtype/device. Sum both
        binary loss terms per valid target position, then average over nonzero target IDs only;
        do not divide by two. Empty/all-padding targets give a differentiable zero connected to
        both logit inputs, with zero gradients. Keep both valid-logit gradients and leave inputs
        unchanged.
    """
    _check_loss(positive, negative, target_ids)
    raise NotImplementedError('Implement sampled_next_item_loss')


# Provided checks: learners implement only the two functions above.
def _check_ids(item_ids):
    """Provided input validation.

    Returns:
        Provided validator: return Python None when item_ids is a nonnegative int64 Tensor
        [B,T], including empty dimensions. Raise ValueError when a checked condition fails; do
        not mutate inputs or return transformed tensors.
    """
    if item_ids.ndim != 2 or item_ids.dtype != torch.long or bool((item_ids < 0).any()):
        raise ValueError('Expected nonnegative int64 [B,T] item IDs; zero means padding')


def _check_loss(positive, negative, target_ids):
    """Provided input validation.

    Returns:
        Provided validator: return Python None when nonnegative int64 target IDs and matching
        floating logits share [B,T] shape/device and logits are finite at valid target
        positions; ignored positions may contain NaN/Inf. Raise ValueError when a checked
        condition fails; do not mutate inputs or return transformed tensors.
    """
    _check_ids(target_ids)
    if positive.shape != target_ids.shape or negative.shape != target_ids.shape:
        raise ValueError('Logits and target IDs must share [B,T] shape')
    if positive.dtype not in (torch.float32, torch.float64) or negative.dtype != positive.dtype:
        raise ValueError('Positive and negative logits must share float32/float64 dtype')
    if positive.device != negative.device or positive.device != target_ids.device:
        raise ValueError('Logits and IDs must share a device')
    valid = target_ids != 0
    if not bool(torch.isfinite(positive[valid]).all()) or not bool(torch.isfinite(negative[valid]).all()):
        raise ValueError('Valid-position logits must be finite')

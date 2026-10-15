# BUG: Exclude items observed by anyone instead of only this user.
"""BPR-MF: observed/unobserved preferences and a global Gaussian-prior penalty."""
import math
import torch
from torch import Tensor
from torch.nn import functional as F


def eligible_negatives(interactions: Tensor, user_ids: Tensor) -> Tensor:
    """Return [B,I] availability: exclude every observed item for each user."""
    _check_mask(interactions, user_ids)
    return (~interactions.any(0)).unsqueeze(0).expand(len(user_ids), -1)


def bpr_mf_loss(user_factors: Tensor, item_factors: Tensor, triples: Tensor, l2: float = 0.) -> Tensor:
    """Mean pairwise negative log likelihood plus l2 times both full table norms."""
    _check_loss(user_factors, item_factors, triples, l2)
    users = user_factors[triples[:, 0]]
    positive = item_factors[triples[:, 1]]
    negative = item_factors[triples[:, 2]]
    margin = (users * (positive - negative)).sum(-1)
    likelihood = F.softplus(-margin).sum() / max(1, len(triples))
    prior = l2 * (user_factors.square().sum() + item_factors.square().sum()) if l2 else 0.
    return likelihood + prior


# Provided validation: implement only the two functions above.
def _check_mask(interactions, user_ids):
    if interactions.ndim != 2 or interactions.dtype != torch.bool:
        raise ValueError('interactions must be Boolean [U,I]')
    if user_ids.ndim != 1 or user_ids.dtype != torch.long or user_ids.device != interactions.device:
        raise ValueError('user_ids must be int64 [B] on the interaction device')
    if bool(((user_ids < 0) | (user_ids >= interactions.shape[0])).any()):
        raise ValueError('User ID is outside the interaction table')


def _check_loss(user_factors, item_factors, triples, l2):
    if user_factors.ndim != 2 or item_factors.ndim != 2 or user_factors.shape[1] < 1 or item_factors.shape[1] != user_factors.shape[1]:
        raise ValueError('Expected user [U,D] and item [I,D] tables with D > 0')
    if user_factors.dtype not in (torch.float32, torch.float64) or item_factors.dtype != user_factors.dtype:
        raise ValueError('Factor tables must share float32/float64 dtype')
    if triples.ndim != 2 or triples.shape[1] != 3 or triples.dtype != torch.long:
        raise ValueError('triples must be int64 [B,3] rows (user,positive,negative)')
    if user_factors.device != item_factors.device or triples.device != user_factors.device:
        raise ValueError('Tables and triples must share a device')
    if not math.isfinite(l2) or l2 < 0:
        raise ValueError('l2 must be finite and nonnegative')
    for column, size in ((0, len(user_factors)), (1, len(item_factors)), (2, len(item_factors))):
        if bool(((triples[:, column] < 0) | (triples[:, column] >= size)).any()):
            raise ValueError('A triple index is outside its factor table')
    if bool((triples[:, 1] == triples[:, 2]).any()):
        raise ValueError('Positive and negative item IDs must differ')
    if not bool(torch.isfinite(user_factors).all()) or not bool(torch.isfinite(item_factors).all()):
        raise ValueError('Factor tables must be finite')

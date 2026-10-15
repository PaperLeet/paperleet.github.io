# BUG: Average document token similarities instead of selecting the maximum.
import torch
from torch import Tensor
from torch.nn.functional import normalize


def maxsim_scores(query: Tensor, documents: Tensor, query_mask: Tensor | None = None,
                  document_mask: Tensor | None = None) -> Tensor:
    """[B,Q,D] x [N,L,D] -> [B,N]; True masks keep encoded token vectors."""
    query_mask, document_mask = _checked_masks(query, documents, query_mask, document_mask)
    q = normalize(query, p=2, dim=-1, eps=1e-12)
    d = normalize(documents, p=2, dim=-1, eps=1e-12)
    similarities = q[:, None] @ d[None].transpose(-1, -2)
    similarities = similarities.masked_fill(~document_mask[None, :, None, :], -torch.inf)
    available = document_mask.any(dim=-1)
    similarities = torch.where(available[None, :, None, None], similarities, torch.zeros_like(similarities))
    best = similarities.mean(dim=-1)
    scores = best.masked_fill(~query_mask[:, None, :], 0.).sum(dim=-1)
    return scores.masked_fill(~available[None, :], -torch.inf)


def pairwise_loss(positive: Tensor, negative: Tensor) -> Tensor:
    """Mean negative log-softmax probability of the positive in each finite pair."""
    _check_pairs(positive, negative)
    margin = negative - positive
    return torch.logaddexp(torch.zeros_like(margin), margin).mean()


# Provided validation. Query augmentation vectors are kept when their mask is True.
def _checked_masks(query, documents, query_mask, document_mask):
    if query.ndim != 3 or documents.ndim != 3 or query.shape[-1] != documents.shape[-1]:
        raise ValueError('Expected [B,Q,D] queries and [N,L,D] documents')
    if query.shape[-1] < 1 or documents.shape[1] < 1:
        raise ValueError('D and physical document length L must be positive')
    if query.dtype not in (torch.float32, torch.float64) or documents.dtype != query.dtype or documents.device != query.device:
        raise ValueError('Inputs must share floating dtype and device')
    result = []
    for value, mask in ((query, query_mask), (documents, document_mask)):
        if mask is None:
            mask = torch.ones(value.shape[:-1], dtype=torch.bool, device=value.device)
        if mask.shape != value.shape[:-1] or mask.dtype != torch.bool or mask.device != value.device:
            raise ValueError('Masks must be Boolean token-shape tensors on the input device')
        result.append(mask)
    return tuple(result)


def _check_pairs(positive, negative):
    if positive.ndim != 1 or positive.shape != negative.shape or positive.numel() < 1:
        raise ValueError('Expected nonempty equal [B] score vectors')
    if positive.dtype not in (torch.float32, torch.float64) or negative.dtype != positive.dtype or negative.device != positive.device:
        raise ValueError('Scores must share floating dtype and device')

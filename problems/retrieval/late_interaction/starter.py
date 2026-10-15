"""Implement ColBERT cosine MaxSim and finite paired ranking loss. Validators are supplied."""
import torch
from torch import Tensor
from torch.nn.functional import normalize

def maxsim_scores(query: Tensor, documents: Tensor, query_mask: Tensor | None = None,
                  document_mask: Tensor | None = None) -> Tensor:
    """All-pairs [B,Q,D] x [N,L,D] -> [B,N]. Normalize tokens, max over valid document tokens, sum kept query tokens.
    True masks keep encoded tokens, including any query-augmentation positions.
    All-masked documents score -inf; empty/fully masked queries contribute zero.
    Preserve dtype/device and gradients to both encodings; exact ties use first document index.
    """
    raise NotImplementedError('Implement maxsim_scores')

def pairwise_loss(positive: Tensor, negative: Tensor) -> Tensor:
    """Finite nonempty [B] pairs -> scalar mean negative log-softmax of the positive. Use stable arithmetic."""
    raise NotImplementedError('Implement pairwise_loss')

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

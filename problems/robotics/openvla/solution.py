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

def _edges(edges, vocab_size):
    _float(edges,1)
    if len(edges)<2 or not isinstance(vocab_size,int) or vocab_size<len(edges) or not (edges[1:]>edges[:-1]).all() or edges[0].item()!=-1. or edges[-1].item()!=1.: raise ValueError("invalid quantizer edges or vocabulary")


def encode_actions(actions: Tensor, edges: Tensor, vocab_size: int) -> Tensor:
    """Return reversed vocabulary IDs with the official right-open digitize convention."""
    _float(actions);_edges(edges,vocab_size)
    if actions.dtype!=edges.dtype or actions.device!=edges.device: raise ValueError("edges must match actions")
    digits=(actions.clamp(-1,1).unsqueeze(-1)>=edges).sum(dim=-1)
    return vocab_size-digits


def decode_actions(tokens: Tensor, edges: Tensor, vocab_size: int) -> Tensor:
    """Decode valid vocabulary IDs to clipped action-bin centers."""
    _edges(edges,vocab_size)
    if tokens.numel()==0 or tokens.dtype!=torch.int64 or tokens.device!=edges.device or ((tokens<0)|(tokens>=vocab_size)).any(): raise ValueError("invalid token IDs")
    centers=(edges[:-1]+edges[1:])/2
    indices=(vocab_size-tokens-1).clamp(0,len(centers)-1)
    return centers[indices]


def unnormalize_actions(actions: Tensor, q01: Tensor, q99: Tensor, mask: Tensor) -> Tensor:
    """Apply quantile scaling only to masked action coordinates; preserve gradients."""
    _float(actions,2);_float(q01,1);_same(q01,q99)
    if q01.shape!=actions.shape[1:] or q01.dtype!=actions.dtype or q01.device!=actions.device or not (q99>q01).all() or mask.shape!=q01.shape or mask.dtype!=torch.bool or mask.device!=actions.device: raise ValueError("invalid quantile statistics")
    physical=.5*(actions+1)*(q99-q01)+q01
    return torch.where(mask,physical,actions)

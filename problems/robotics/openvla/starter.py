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
    'Return reversed vocabulary IDs with the official right-open digitize convention.'
    raise NotImplementedError("Implement encode_actions according to the README contract")


def decode_actions(tokens: Tensor, edges: Tensor, vocab_size: int) -> Tensor:
    'Decode valid vocabulary IDs to clipped action-bin centers.'
    raise NotImplementedError("Implement decode_actions according to the README contract")


def unnormalize_actions(actions: Tensor, q01: Tensor, q99: Tensor, mask: Tensor) -> Tensor:
    'Apply quantile scaling only to masked action coordinates; preserve gradients.'
    raise NotImplementedError("Implement unnormalize_actions according to the README contract")

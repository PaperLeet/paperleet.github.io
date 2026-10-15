import math
import torch
from torch import Tensor


def block_diffusion_mask(length: int, block_size: int, device=None) -> Tensor:
    """Allowed edges for concatenated [noisy tokens, clean tokens].

    Returns:
        Return one torch.bool Tensor of shape [2L, 2L], where L=length. Rows are queries and
        columns are keys; both axes concatenate L noisy tokens then L clean tokens. True
        means allowed. Use the requested device (CPU by default in this exercise); the mask
        has no gradients and is not an additive floating mask.
    """
    if type(length) is not int or type(block_size) is not int or length<=0 or block_size<=0:
        raise ValueError("length and block_size must be positive integers")
    raise NotImplementedError("Build the four attention-mask quadrants")


def block_diffusion_attention(query: Tensor, key: Tensor, value: Tensor, block_size: int) -> Tensor:
    """Dense CPU attention using the vectorized block-diffusion training mask.

    Returns:
        Return the attention output Tensor [B, H, 2L, Dv], matching value: B is batch size,
        H heads, L tokens per stream, and Dv value width. Return aggregated features, not
        the [B,H,2L,2L] attention weights. Preserve input float32/float64 dtype, device and
        gradients through Q/K/V; do not mutate inputs.
    """
    if query.ndim!=4 or query.numel()==0 or key.shape!=query.shape or query.shape[-2]%2:
        raise ValueError("Q/K must have equal nonempty [B,H,2L,D] shapes")
    if value.ndim!=4 or value.shape[:-1]!=query.shape[:-1] or value.shape[-1]==0:
        raise ValueError("V must have shape [B,H,2L,Dv] with Dv>0")
    raise NotImplementedError("Apply the mask before normalizing scores")

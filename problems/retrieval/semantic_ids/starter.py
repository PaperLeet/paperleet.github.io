import torch
from torch import Tensor


def residual_quantize(latents: Tensor, codebooks: Tensor) -> tuple[Tensor, Tensor]:
    """Required: finite CPU float32/64 latents [B,D] and books [L,K,D]. B>=0; L,K,D>=1. Use each
    level's nearest vector to the remaining residual, lowest index on ties. Return int64 codes
    [B,L] and summed vectors [B,D]. Preserve floating dtype/device, freeze outputs, and do not
    mutate inputs. Inputs may be noncontiguous; no gradients/training are required.

    Returns:
        Return (codes, reconstruction), a Python tuple of two Tensors. codes is int64 [B,L] (B
        items, L quantization levels); reconstruction is detached floating [B,D], the sum of all
        selected level vectors, with the latent dtype/device. Codes share that device and choose
        the lowest index on exact distance ties. B=0 preserves both empty shapes. Neither output
        requires gradients; do not mutate inputs.
    """
    raise NotImplementedError()


def resolve_collisions(codes: Tensor) -> Tensor:
    """Required: nonnegative CPU int64 [B,L], B>=0 and L>=1. Append an int64 ordinal column: count
    previous identical full rows. Thus each group gets 0,1,... in input occurrence order;
    singletons get 0. Return [B,L+1] on the same device without mutating codes.

    Returns:
        Return one int64 Tensor [B,L+1] on the codes device, preserving the original L code
        columns and appending a zero-based occurrence count within each identical full-row
        group. Keep row order; each singleton suffix is zero, and B=0 returns [0,L+1]. No
        gradients or input mutation.
    """
    raise NotImplementedError()

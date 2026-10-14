import torch
from torch import Tensor


def residual_quantize(latents: Tensor, codebooks: Tensor) -> tuple[Tensor, Tensor]:
    """Required: finite CPU float32/64 latents [B,D] and books [L,K,D].
    B>=0; L,K,D>=1. Use each level's nearest vector to the remaining residual,
    lowest index on ties. Return int64 codes [B,L] and summed vectors [B,D].
    Preserve floating dtype/device, freeze outputs, and do not mutate inputs.
    Inputs may be noncontiguous; no gradients/training are required."""
    raise NotImplementedError()


def resolve_collisions(codes: Tensor) -> Tensor:
    """Required: nonnegative CPU int64 [B,L], B>=0 and L>=1.
    Append an int64 ordinal column: count previous identical full rows.
    Thus each group gets 0,1,... in input occurrence order; singletons get 0.
    Return [B,L+1] on the same device without mutating codes."""
    raise NotImplementedError()

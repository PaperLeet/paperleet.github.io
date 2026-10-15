import torch
import torch.nn.functional as F
from torch import Tensor


def _validate_codes(token_maps, codebook, projections):
    """Provided helper.

    Returns:
        Return a Python `list[int]` of square side lengths, one per token map in input
        order, after validating the scale schedule and tensor layouts. It is not a Tensor.
        Raise ValueError for the checked invalid inputs; do not modify any input.
    """
    if not token_maps or codebook.ndim != 2 or min(codebook.shape) == 0:
        raise ValueError("nonempty token maps and [V,C] codebook required")
    if any(t.ndim != 3 for t in token_maps):
        raise ValueError("token maps must have three dimensions")
    scales = [t.shape[-1] for t in token_maps]
    if scales[0] != 1 or any(a >= b for a, b in zip(scales, scales[1:])):
        raise ValueError("square scales must start at 1 and strictly increase")
    batch = token_maps[0].shape[0]
    for t, n in zip(token_maps, scales):
        if t.ndim != 3 or t.shape != (batch, n, n) or batch == 0 or t.dtype != torch.long:
            raise ValueError("tokens must be nonempty int64 [B,n,n]")
        if t.device != codebook.device or (t < 0).any() or (t >= len(codebook)).any():
            raise ValueError("token indices must be on the codebook device and in range")
    channels = codebook.shape[1]
    if projections.shape != (len(scales), channels, channels):
        raise ValueError("projections must have shape [K,C,C]")
    if projections.dtype != codebook.dtype or projections.device != codebook.device:
        raise ValueError("projection dtype/device must match codebook")
    return scales


def lift_codes(tokens: Tensor, codebook: Tensor, projection: Tensor, full_size: int) -> Tensor:
    """Provided: code lookup, bicubic upsampling, then a fixed channel projection.

    Returns:
        Return one floating Tensor `[B,C,F,F]`, where B is batch size, C the projected
        channel width and F=full_size. It contains codebook features lifted to full spatial
        resolution and channel-projected, in channels-first layout. Preserve the shared
        codebook/projection dtype/device and input values. This helper itself does not
        detach gradients from codebook/projection; next_scale_inputs calls it in a no-grad
        context.
    """
    codes = F.embedding(tokens, codebook).permute(0, 3, 1, 2)
    if codes.shape[-1] != full_size:
        codes = F.interpolate(codes, size=(full_size, full_size), mode="bicubic", align_corners=False)
    return torch.einsum("oc,bchw->bohw", projection, codes)


def scale_causal_mask(scales: list[int], device=None) -> Tensor:
    """Return [L,L] bool visibility: keys at the same or an earlier scale.

    Returns:
        Return one `torch.bool` Tensor `[L,L]` on the requested device, with L=sum(n*n for n
        in scales). Rows are query tokens and columns key tokens, concatenated by increasing
        scale and raster order within each scale. True allows the same or an earlier scale,
        including every within-scale pair. It is a visibility matrix without gradients, not
        an additive floating-point mask.
    """
    if not scales or any(type(n) is not int or n <= 0 for n in scales):
        raise ValueError("scales must contain positive integers")
    if scales[0] != 1 or any(a >= b for a, b in zip(scales, scales[1:])):
        raise ValueError("scales must start at 1 and strictly increase")
    # TODO: implement the core algorithm.
    raise NotImplementedError


@torch.no_grad()
def next_scale_inputs(token_maps: list[Tensor], codebook: Tensor, projections: Tensor) -> Tensor:
    """Frozen tokenizer features for scales 2..K; start embedding is provided separately.

    Returns:
        Return one floating Tensor `[B,L_next,C]`: B batch size, C codebook channel width
        and L_next=sum(n*n for n in scales[1:]). Concatenate features for target scales 2..K
        in scale order, with row-major spatial tokens within each scale; do not include the
        separate first-scale start embedding. Preserve codebook dtype/device. A single scale
        returns `[B,0,C]`. The frozen-tokenizer result must have requires_grad=False even if
        codebook/projections require gradients; do not mutate inputs.
    """
    scales = _validate_codes(token_maps, codebook, projections)
    # TODO: implement the core algorithm.
    raise NotImplementedError

import math
import torch
from torch import Tensor


def sinusoidal_encoding(length: int, d_model: int, start: int = 0, *,
                        dtype=torch.float32, device=None) -> Tensor:
    """Return [length, d_model]; positions start..start+length-1, odd widths allowed.

    Returns:
        Return one floating Tensor `[length,d_model]`, without a batch dimension. Row i
        encodes absolute position start+i; columns are feature channels, with sine in even
        channels and cosine in odd channels. Honor the requested float32/float64 dtype and
        device. Odd widths retain the final sine channel; length=0 returns `[0,d_model]`.
        These fixed encodings have requires_grad=False; return the Tensor rather than a list
        or a pair of tables.
    """
    raise NotImplementedError("Implement sinusoidal_encoding")

import math
import torch
from torch import Tensor
import torch.nn.functional as F


def _floating(*values):
    first = values[0]
    for value in values:
        if not value.is_floating_point() or value.dtype != first.dtype or value.device != first.device or not torch.isfinite(value).all():
            raise ValueError("floating inputs must be finite and share dtype/device")

def window_permutation(frames: int, height: int, width: int, merge_size: int, window_size: int) -> tuple[Tensor, Tensor]:
    """All arguments are positive integers. height/width describe the original patch grid and are divisible by merge_size; window_size counts merged cells. Iterate frames, window rows/columns, merged-cell rows/columns within a window, then patch rows/columns within a cell. Return original row-major patch indices without boundary padding. Outputs are CPU int64 permutation [T*H*W] and cu_seqlens counted in original patches, starting at zero."""
    raise NotImplementedError("Implement window_permutation")

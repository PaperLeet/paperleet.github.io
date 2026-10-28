import math
import torch
from torch import Tensor
import torch.nn.functional as F


def _floating(*values):
    first = values[0]
    for value in values:
        if not value.is_floating_point() or value.dtype != first.dtype or value.device != first.device or not torch.isfinite(value).all():
            raise ValueError("floating inputs must be finite and share dtype/device")

def time_aligned_interleave(video_times: Tensor, audio_times: Tensor) -> tuple[Tensor, Tensor]:
    """video_times [V] and audio_times [A] are separately nondecreasing, nonnegative floating seconds, with at least one nonempty stream and shared dtype/device. Chunks are half-open [2c,2c+2). Order chunks increasingly; place all video before all audio within each chunk, preserving each stream’s order. Return int64 permutation into concat(video,audio) and reordered temporal_ids=floor(time*25). Spatial H/W axes are excluded; quantized times and indices are not differentiable."""
    raise NotImplementedError("Implement time_aligned_interleave")

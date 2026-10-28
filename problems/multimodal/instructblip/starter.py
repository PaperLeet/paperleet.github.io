import math
import torch
from torch import Tensor
import torch.nn.functional as F


def _floating(*values):
    first = values[0]
    for value in values:
        if not value.is_floating_point() or value.dtype != first.dtype or value.device != first.device or not torch.isfinite(value).all():
            raise ValueError("floating inputs must be finite and share dtype/device")

def instruction_query_block(queries: Tensor, instruction: Tensor, visual: Tensor, instruction_mask: Tensor) -> Tensor:
    """queries is [Q,D], instruction [I,D], visual [N,D], and instruction_mask boolean [I]. First let queries attend to themselves and valid instruction tokens, with a residual. Then use updated queries for visual cross-attention, with another residual. Identity projections, one head, no LayerNorm/FFN define this reduced teaching block. All-masked instructions still leave query self-attention valid."""
    raise NotImplementedError("Implement instruction_query_block")

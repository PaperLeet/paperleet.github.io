import math
import torch
from torch import Tensor

def parallel_mlp(x: Tensor, w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor, parts: int) -> Tensor:
    """(N,O) output with autograd through every floating input.

x:(N,D), w1:(D,H), b1:(H,), w2:(H,O), b2:(O,). H is divisible by parts. Partition first-layer columns and corresponding second-layer rows. Add b2 once after reduction. Preserve gradients, dtype and input values."""
    raise NotImplementedError()

import torch
from torch import Tensor

def zero2_adam_step(params: Tensor, gradients: Tensor, first: Tensor, second: Tensor, step: int, lr: float, beta1: float=0.9, beta2: float=0.999, eps: float=1e-08) -> tuple[Tensor, Tensor, Tensor]:
    """New sharded parameters and Adam moments, without input mutation.

Parameters and moments have shape (R,S); replicated gradients have shape (R,R*S). Return updated parameter shards and both moments. Average across data-parallel ranks; step starts at one and weight decay is absent. Byte accounting uses two-byte weights, two-byte gradients, and twelve bytes for master weights plus two moments. Parameter count is divisible by R."""
    raise NotImplementedError()

def state_bytes(parameters: int, world_size: int, stage: int) -> int:
    """Per-rank model-state bytes for ZeRO stage 0, 1, 2 or 3.

Parameters and moments have shape (R,S); replicated gradients have shape (R,R*S). Return updated parameter shards and both moments. Average across data-parallel ranks; step starts at one and weight decay is absent. Byte accounting uses two-byte weights, two-byte gradients, and twelve bytes for master weights plus two moments. Parameter count is divisible by R."""
    raise NotImplementedError()

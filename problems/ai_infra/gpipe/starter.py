import torch
from torch import Tensor

def pipeline_schedule(stages: int, microbatches: int) -> list[tuple]:
    """Unit-duration forward/backward event records.

stages, microbatches and chunk_size are positive integers. Events are (phase, stage, microbatch, start), with phase F/B and unit duration. All forward events finish before reverse-order backward events. x is (N,D), y (N,), w (D,); a short final chunk is allowed. Divide total squared error by N."""
    raise NotImplementedError()

def microbatch_mse(x: Tensor, y: Tensor, weight: Tensor, chunk_size: int) -> Tensor:
    """Differentiable scalar sample-mean squared error.

stages, microbatches and chunk_size are positive integers. Events are (phase, stage, microbatch, start), with phase F/B and unit duration. All forward events finish before reverse-order backward events. x is (N,D), y (N,), w (D,); a short final chunk is allowed. Divide total squared error by N."""
    raise NotImplementedError()

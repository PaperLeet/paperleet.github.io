import torch
from torch import Tensor


def clip_gradients(gradients: list[Tensor], max_norm: float, eps: float = 1e-6) -> tuple[list[Tensor], Tensor]:
    raise NotImplementedError

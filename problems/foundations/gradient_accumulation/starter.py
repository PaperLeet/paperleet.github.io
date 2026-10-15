import torch
from torch import Tensor


def accumulate_gradients(gradients: list[Tensor], token_counts: list[int]) -> Tensor:
    raise NotImplementedError

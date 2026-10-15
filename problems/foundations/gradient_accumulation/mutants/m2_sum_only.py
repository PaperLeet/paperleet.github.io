# BUG: 加权求和后未除全局 token 数。
import torch
from torch import Tensor

def accumulate_gradients(gradients: list[Tensor], token_counts: list[int]) -> Tensor:
    total = sum(token_counts)
    result = torch.zeros_like(gradients[0])
    for gradient, count in zip(gradients, token_counts):
        if count:
            result = result + gradient * (count)
    return result

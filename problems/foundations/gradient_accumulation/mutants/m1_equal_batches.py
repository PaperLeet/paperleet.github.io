# BUG: 等权平均不同 token 数的 microbatch。
import torch
from torch import Tensor

def accumulate_gradients(gradients: list[Tensor], token_counts: list[int]) -> Tensor:
    total = sum(token_counts)
    result = torch.zeros_like(gradients[0])
    for gradient, count in zip(gradients, token_counts):
        if count:
            result = result + gradient * (1 / len(gradients))
    return result

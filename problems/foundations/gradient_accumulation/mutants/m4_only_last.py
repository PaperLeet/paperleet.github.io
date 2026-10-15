# BUG: 只返回最后一个 microbatch 梯度。
import torch
from torch import Tensor

def accumulate_gradients(gradients: list[Tensor], token_counts: list[int]) -> Tensor:
    total = sum(token_counts)
    result = torch.zeros_like(gradients[0])
    for gradient, count in zip(gradients, token_counts):
        if count:
            result = result + gradient * (count / total)
    return gradients[-1].clone()

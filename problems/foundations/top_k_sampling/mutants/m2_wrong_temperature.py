# BUG: 将温度乘到 logits 上。
import torch
from torch import Tensor

def top_k_probs(logits: Tensor, k: int, temperature: float = 1.) -> Tensor:
    order = torch.argsort(logits, dim=-1, descending=True, stable=True)
    ranks = torch.argsort(order, dim=-1)
    temperature = 1. / temperature
    work = logits.to(torch.float64)
    if temperature >= 1.:
        scaled = work / temperature
        scaled = scaled - scaled.amax(dim=-1, keepdim=True)
    else:
        scaled = (work - work.amax(dim=-1, keepdim=True)) / temperature
    return torch.softmax(scaled.masked_fill(ranks >= k, float('-inf')), dim=-1).to(logits.dtype)

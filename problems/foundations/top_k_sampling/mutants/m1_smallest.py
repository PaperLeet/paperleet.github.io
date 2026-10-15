# BUG: 保留最小的 k 个 logit。
import torch
from torch import Tensor

def top_k_probs(logits: Tensor, k: int, temperature: float = 1.) -> Tensor:
    order = torch.argsort(logits, dim=-1, descending=False, stable=True)
    ranks = torch.argsort(order, dim=-1)
    work = logits.to(torch.float64)
    if temperature >= 1.:
        scaled = work / temperature
        scaled = scaled - scaled.amax(dim=-1, keepdim=True)
    else:
        scaled = (work - work.amax(dim=-1, keepdim=True)) / temperature
    return torch.softmax(scaled.masked_fill(ranks >= k, float('-inf')), dim=-1).to(logits.dtype)

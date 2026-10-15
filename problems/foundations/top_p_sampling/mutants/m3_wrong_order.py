# BUG: 输出仍按概率排序，未还原词表。
import torch
from torch import Tensor

def top_p_probs(logits: Tensor, p: float, temperature: float = 1.) -> Tensor:
    # Python floats must not be narrowed to float32 (tiny p/T would become 0).
    work = logits.to(torch.float64)
    if temperature >= 1.:
        scaled = work / temperature
        scaled = scaled - scaled.amax(dim=-1, keepdim=True)
    else:
        scaled = (work - work.amax(dim=-1, keepdim=True)) / temperature
    probs = torch.softmax(scaled, dim=-1)
    order = torch.argsort(probs, dim=-1, descending=True, stable=True)
    sorted_probs = probs.gather(-1, order)
    previous = torch.cat((torch.zeros_like(sorted_probs[..., :1]), sorted_probs.cumsum(-1)[..., :-1]), dim=-1)
    keep_sorted = torch.ones_like(previous, dtype=torch.bool) if p == 1. else previous < p
    keep = keep_sorted
    return torch.softmax(scaled.masked_fill(~keep, float('-inf')), dim=-1).to(logits.dtype)

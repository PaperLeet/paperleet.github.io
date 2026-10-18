# Ignores the caller-owned generator.
import torch
from torch import Tensor

def nucleus_probs(logits: Tensor, p: float, temperature: float = 1.) -> Tensor:
    """Provided: float64 nucleus probabilities in original vocabulary order."""
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
    keep = keep_sorted.gather(-1, torch.argsort(order, dim=-1))
    return torch.softmax(scaled.masked_fill(~keep, float('-inf')), dim=-1)


def sample_top_p(logits: Tensor, p: float, temperature: float, generator: torch.Generator) -> Tensor:
    """Draw one token per row, consuming one float64 uniform per row."""
    probs = nucleus_probs(logits, p, temperature)
    cdf = probs.cumsum(dim=-1)
    cdf = cdf / cdf[..., -1:]
    uniforms = torch.rand(logits.shape[:-1], dtype=torch.float64)
    return (cdf <= uniforms.unsqueeze(-1)).sum(dim=-1).to(torch.int64)

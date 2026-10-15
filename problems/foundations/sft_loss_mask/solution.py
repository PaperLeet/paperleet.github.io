import torch
from torch import Tensor

def sft_loss(logits: Tensor, tokens: Tensor, response_mask: Tensor, attention_mask: Tensor) -> Tensor:
    scores = logits[:, :-1, :].to(torch.float64)
    scores = scores - scores.amax(dim=-1, keepdim=True)
    targets = tokens[:, 1:]
    valid = response_mask[:, 1:] & attention_mask[:, 1:]
    correct = scores.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    losses = torch.logsumexp(scores, dim=-1) - correct
    numerator = torch.where(valid, losses, torch.zeros_like(losses)).sum()
    return (numerator / valid.sum().clamp(min=1)).to(logits.dtype)

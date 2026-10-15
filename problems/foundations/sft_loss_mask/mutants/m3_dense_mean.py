# BUG: 除以所有位置数，而非有效 token 数。
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
    return (numerator / max(1, losses.numel())).to(logits.dtype)

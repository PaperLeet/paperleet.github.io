import torch
from torch import Tensor

def next_token_loss(logits: Tensor, tokens: Tensor) -> Tensor:
    if logits.shape[1] == 1:
        return (logits * 0).sum()
    scores = logits[:, :-1, :].to(torch.float64)
    scores = scores - scores.amax(dim=-1, keepdim=True)
    targets = tokens[:, 1:]
    correct = scores.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (torch.logsumexp(scores, dim=-1) - correct).mean().to(logits.dtype)

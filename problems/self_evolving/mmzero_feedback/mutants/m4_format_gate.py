# BUG: Gate agreement instead of adding format reward.
import torch
from torch import Tensor

def coder_rewards(executed: Tensor, easy_accuracy: Tensor, consistency: Tensor, syntax_error: Tensor) -> Tensor:
    """bool executed/syntax_error [B], float easy_accuracy/consistency [B] in [0,1].
    Successful score=1+easy+min(c,1-c); failed score=-.1. Additionally subtract
    .05 for syntax_error (both error penalties may apply). Unrendered visual
    scores are ignored. Return detached same-dtype [B]. Uses Eq.7's continuous
    difficulty score; Eq.11's later binary range annotation is inconsistent.
    """
    score = 1 + easy_accuracy.detach() + torch.minimum(consistency.detach(), 1 - consistency.detach())
    return torch.where(executed, score, torch.full_like(score, -.1)) - .05 * syntax_error.to(score.dtype)

def solver_rewards(answers: Tensor, format_valid: Tensor, num_answers: int, alpha: float = .9) -> Tensor:
    """Long answer IDs [B,K], bool format_valid [B,K], B,K>=1, 0<=alpha<=1.
    Answer IDs are normalized upstream and lie in [0,num_answers). Majority
    includes ALL K answers; tie chooses smallest ID. Return float64 [B,K]:
    alpha*agreement + (1-alpha)*format_valid. Format is an ADDITIVE component,
    not a gate. Consensus is a silver label, not an independent gold answer.
    """
    counts = (answers[:, :, None] == torch.arange(num_answers, device=answers.device)).sum(dim=1)
    labels = counts.argmax(dim=1)
    return alpha * (answers == labels[:, None]).to(torch.float64) * format_valid.to(torch.float64)

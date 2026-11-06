# BUG: Detach the selected logits.
"""BEiT's masked discrete visual-token classification objective."""
import torch
from torch import Tensor


def masked_token_cross_entropy(logits: Tensor, token_ids: Tensor, mask: Tensor) -> Tensor:
    """Mean stable CE over masked [B,N] positions of [B,N,K] logits; empty masks give zero."""
    _check(logits, token_ids, mask)
    selected = logits[mask].detach()
    labels = token_ids[mask]
    if selected.shape[0] == 0:
        return logits.sum() * 0
    target_scores = selected.gather(1, labels[:, None]).squeeze(1)
    return (selected.logsumexp(-1) - target_scores).mean()


# Provided validation. Unselected token IDs are ignored, including sentinel values.
def _check(logits, token_ids, mask):
    if logits.ndim != 3 or min(logits.shape) < 1 or logits.dtype not in (torch.float32, torch.float64):
        raise ValueError('logits must be nonempty float32/float64 [B,N,K]')
    if not bool(torch.isfinite(logits).all()):
        raise ValueError('logits must be finite')
    if token_ids.shape != logits.shape[:2] or token_ids.dtype != torch.int64 or token_ids.device != logits.device:
        raise ValueError('token IDs must be int64 [B,N] on the logits device')
    if mask.shape != logits.shape[:2] or mask.dtype != torch.bool or mask.device != logits.device:
        raise ValueError('mask must be boolean [B,N] on the logits device')
    labels = token_ids[mask]
    if bool((labels < 0).any()) or bool((labels >= logits.shape[-1]).any()):
        raise ValueError('masked token IDs must be in [0,K)')

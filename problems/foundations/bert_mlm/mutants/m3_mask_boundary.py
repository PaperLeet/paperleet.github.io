# BUG: Put exactly .8 in the MASK branch
import torch


def make_mlm_inputs(tokens, selected, draws, random_ids, mask_token_id, ignore_index=-100):
    if tokens.ndim != 2 or any(a.shape != tokens.shape for a in (selected, draws, random_ids)):
        raise ValueError("matching [B,L] arrays required")
    if tokens.dtype != torch.int64 or random_ids.dtype != torch.int64 or selected.dtype != torch.bool:
        raise ValueError("int64 tokens/replacements and boolean selection required")
    if draws.dtype not in (torch.float32, torch.float64) or not torch.isfinite(draws).all() or ((draws < 0) | (draws >= 1)).any():
        raise ValueError("draws must lie in [0,1)")
    if (tokens < 0).any() or (random_ids < 0).any() or not isinstance(mask_token_id, int) or mask_token_id < 0 or not isinstance(ignore_index, int) or ignore_index >= 0:
        raise ValueError("nonnegative token IDs and negative ignore index required")
    replace_mask = selected & (draws <= .8)
    replace_random = selected & (draws >= .8) & (draws < .9)
    corrupted = torch.where(replace_mask, torch.full_like(tokens, mask_token_id), torch.where(replace_random, random_ids, tokens))
    labels = torch.where(selected, tokens, torch.full_like(tokens, ignore_index))
    return corrupted, labels


def masked_lm_loss(logits, labels, ignore_index=-100):
    if logits.ndim != 3 or labels.shape != logits.shape[:2] or logits.shape[2] < 1:
        raise ValueError("logits [B,L,V], labels [B,L], V>0 required")
    if logits.dtype not in (torch.float32, torch.float64) or labels.dtype != torch.int64 or not isinstance(ignore_index, int) or ignore_index >= 0:
        raise ValueError("floating logits, int64 labels and negative ignore index required")
    active = labels != ignore_index
    if ((labels[active] < 0) | (labels[active] >= logits.shape[2])).any():
        raise ValueError("target ID outside prediction vocabulary")
    if not active.any():
        return logits.sum() * 0.
    safe = labels.clamp(min=0)
    logp = torch.log_softmax(logits, dim=-1)
    nll = -torch.gather(logp, -1, safe.unsqueeze(-1)).squeeze(-1)
    return (nll * active.to(logits.dtype)).sum() / active.sum()

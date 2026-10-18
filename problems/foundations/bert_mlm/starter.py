from __future__ import annotations
import torch


def make_mlm_inputs(tokens: torch.Tensor, selected: torch.Tensor, draws: torch.Tensor, random_ids: torch.Tensor, mask_token_id: int, ignore_index: int = -100) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Implement 80/10/10 selected-token corruption, original-token labels and stable cross entropy averaged only over selected positions.

    For `make_mlm_inputs`, all four arrays have shape `[B,L]`: nonnegative int64 tokens/random_ids, bool selected, and finite float32/float64 draws in `[0,1)`. Return independent int64 `(corrupted,labels)` arrays without modifying inputs. At selected sites use mask_token_id for `u<0.8`, random_ids for `0.8<=u<0.9`, and the original token otherwise. Unselected tokens stay unchanged and receive the negative integer ignore_index (default -100). Selected labels always hold the original token; **do not shift targets**. The nonnegative integer mask_token_id may lie outside the prediction vocabulary. The caller handles 15% selection, special-token exclusion and replacement sampling.

    `masked_lm_loss` takes finite float32/float64 logits `[B,L,V]`, `V>0`, and int64 labels `[B,L]` on the same device. Targets must be ignore_index or in `[0,V)`. Return a scalar in the logit dtype, averaged over non-ignored sites. With no targets, including an empty batch/sequence, return graph-connected zero with zero logit gradients. Invalid shapes, dtypes, draws or target IDs raise ValueError.
    """
    raise NotImplementedError

def masked_lm_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """
    Implement 80/10/10 selected-token corruption, original-token labels and stable cross entropy averaged only over selected positions.

    For `make_mlm_inputs`, all four arrays have shape `[B,L]`: nonnegative int64 tokens/random_ids, bool selected, and finite float32/float64 draws in `[0,1)`. Return independent int64 `(corrupted,labels)` arrays without modifying inputs. At selected sites use mask_token_id for `u<0.8`, random_ids for `0.8<=u<0.9`, and the original token otherwise. Unselected tokens stay unchanged and receive the negative integer ignore_index (default -100). Selected labels always hold the original token; **do not shift targets**. The nonnegative integer mask_token_id may lie outside the prediction vocabulary. The caller handles 15% selection, special-token exclusion and replacement sampling.

    `masked_lm_loss` takes finite float32/float64 logits `[B,L,V]`, `V>0`, and int64 labels `[B,L]` on the same device. Targets must be ignore_index or in `[0,V)`. Return a scalar in the logit dtype, averaged over non-ignored sites. With no targets, including an empty batch/sequence, return graph-connected zero with zero logit gradients. Invalid shapes, dtypes, draws or target IDs raise ValueError.
    """
    raise NotImplementedError

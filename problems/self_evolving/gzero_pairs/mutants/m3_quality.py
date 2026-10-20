# BUG: Ignore structural quality.
import torch
from torch import Tensor

def preference_mask(delta: Tensor, quality: Tensor) -> Tensor:
    """Finite float delta [B], bool quality [B], B>=1. Compute the LINEAR median
    of the full pool first; retain delta<=median AND quality. Include ties and
    negative deltas; a singleton is retained iff quality. Return bool [B].
    Quality checks are upstream. The median is NOT recomputed after filtering.
    """
    ordered = torch.sort(delta.detach()).values
    n = delta.shape[0]
    median = (ordered[(n - 1) // 2] + ordered[n // 2]) / 2
    return (delta <= median)

def build_preferences(records: list[dict], delta: Tensor, quality: Tensor) -> list[dict]:
    """B records have question, hint, unassisted, assisted string fields.
    Return NEW dictionaries in original order for preference_mask-selected rows:
    exactly prompt=question, chosen=assisted, rejected=unassisted. Never put the
    hint into prompt, mutate inputs or include delta/hint in output dictionaries.
    """
    keep = preference_mask(delta, quality).tolist()
    return [dict(prompt=r['question'], chosen=r['assisted'], rejected=r['unassisted'])
            for r, accepted in zip(records, keep) if accepted]

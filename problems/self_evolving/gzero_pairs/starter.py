import torch
from torch import Tensor

def preference_mask(delta: Tensor, quality: Tensor) -> Tensor:
    """Finite float delta [B], bool quality [B], B>=1. Compute the LINEAR median
    of the full pool first; retain delta<=median AND quality. Include ties and
    negative deltas; a singleton is retained iff quality. Return bool [B].
    Quality checks are upstream. The median is NOT recomputed after filtering.
    """
    raise NotImplementedError()

def build_preferences(records: list[dict], delta: Tensor, quality: Tensor) -> list[dict]:
    """B records have question, hint, unassisted, assisted string fields.
    Return NEW dictionaries in original order for preference_mask-selected rows:
    exactly prompt=question, chosen=assisted, rejected=unassisted. Never put the
    hint into prompt, mutate inputs or include delta/hint in output dictionaries.
    """
    raise NotImplementedError()

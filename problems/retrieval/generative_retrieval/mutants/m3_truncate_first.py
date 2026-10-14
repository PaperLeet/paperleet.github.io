# BUG: Truncates before discarding unknown IDs.
from collections.abc import Callable
import torch
from torch import Tensor


def beam_search(next_logits: Callable[[tuple[int, ...]], Tensor], length: int,
                beam_width: int) -> list[tuple[tuple[int, ...], float]]:
    beams = [((), 0.0)]
    for _ in range(length):
        candidates = []
        for prefix, score in beams:
            log_probs = torch.log_softmax(next_logits(prefix), dim=-1).tolist()
            for token, log_prob in enumerate(log_probs):
                candidates.append((prefix + (token,), score + log_prob))
        beams = sorted(candidates, key=lambda item: (-item[1], item[0]))[:beam_width]
    return beams


def lookup_candidates(beams: list[tuple[tuple[int, ...], float]], catalog_ids: Tensor,
                      k: int) -> list[tuple[int, float]]:
    if k == 0:
        return []
    catalog = {tuple(row): index for index, row in enumerate(catalog_ids.tolist())}
    result, seen = [], set()
    for tokens, score in beams[:k]:
        index = catalog.get(tokens)
        if index is None or index in seen:
            continue
        result.append((index, score))
        seen.add(index)
        if len(result) == k:
            break
    return result

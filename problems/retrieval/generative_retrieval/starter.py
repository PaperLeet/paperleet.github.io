from collections.abc import Callable
import torch
from torch import Tensor


def beam_search(next_logits: Callable[[tuple[int, ...]], Tensor], length: int,
                beam_width: int) -> list[tuple[tuple[int, ...], float]]:
    """Required: length>=0, beam_width>=1 (integers). Start with ((),0.0). Callback returns finite
    frozen CPU float64 logits [V], V>=1 fixed. Each step expands ALL retained prefixes over ALL
    V tokens. Sum full-vocab log-softmax probabilities; globally retain at most beam_width
    sequences. Sort by descending score, then lexicographic tokens on ties. Return a list of
    (token tuple, Python float log-score). No EOS or catalog constraints. Do not mutate logits.
    length=0 returns [((),0.0)].

    Returns:
        Return a Python list[tuple[tuple[int, ...], float]]; each entry is (tokens,
        cumulative_log_score), with exactly length Python integer tokens and a Python float
        score, not a Tensor. Keep at most beam_width entries, ordered by descending score then
        lexicographic tokens for exact ties. length=0 returns [((), 0.0)]. This is frozen
        decoding without gradients; do not mutate callback logits.
    """
    raise NotImplementedError()


def lookup_candidates(beams: list[tuple[tuple[int, ...], float]], catalog_ids: Tensor,
                      k: int) -> list[tuple[int, float]]:
    """Required: beams already ranked as above, possibly repeated sequences. catalog_ids is unique
    CPU int64 [N,L]; k>=0. Discard unknown full IDs and repeated items BEFORE taking k results.
    Return (catalog row index, original log-score) pairs in beam order. No renormalization or
    prefix matching. k=0, empty beams/catalog, or no valid IDs returns []; no input mutation.

    Returns:
        Return a Python list[tuple[int, float]] of (catalog_row_index, original_log_score),
        preserving beam order after removing unknown full IDs and duplicate items. The integer
        indexes a row of catalog [N,L], not a token. Return at most k entries; k=0, empty
        input/catalog or no valid IDs gives []. Scores remain Python floats without
        renormalization or gradients; do not mutate inputs.
    """
    raise NotImplementedError()

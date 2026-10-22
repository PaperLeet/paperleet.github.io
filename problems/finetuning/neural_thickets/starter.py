# Contract: finite CPU float32/64 with matching dtype/device; preserve both.
# Support noncontiguous inputs; never mutate inputs. See the statement for full conventions.
import torch
from torch import Tensor


def perturb_candidates(theta: Tensor, noise: Tensor, scales: Tensor) -> Tensor:
    """theta:S, noise:[N,*S], scales:[N]>=0 -> detached [N,*S], N>=1.
    S scalar/nonempty. Return theta+scales_i*noise_i from the same base."""
    raise NotImplementedError()


def select_experts(scores: Tensor, k: int) -> Tensor:
    """Finite scores:[N], 1<=k<=N -> long [k] indices, descending score.
    Score ties favor lower original index; do not use test labels."""
    raise NotImplementedError()


def majority_vote(predictions: Tensor, selected: Tensor, num_classes: int) -> Tensor:
    """Long predictions:[N,B], distinct selected:[K], K>=1 -> long [B].
    Count selected experts equally per sample; classes in [0,num_classes).
    Ties -> smallest class; B=0 -> empty. num_classes>=1."""
    raise NotImplementedError()

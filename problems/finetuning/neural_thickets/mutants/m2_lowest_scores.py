# Bug: Select the worst-scoring experts.
import torch
from torch import Tensor


def perturb_candidates(theta: Tensor, noise: Tensor, scales: Tensor) -> Tensor:
    """Return detached [N,*S] candidates theta+scale_i*noise_i from the same base.

    theta is nonempty shape S (possibly scalar), noise [N,*S], scales [N]>=0;
    N>=1, matching floating dtype/device. Randomness is supplied by the caller.
    No sequential updates, parameter averaging, mutation or autograd graph.
    """
    scale = scales.detach().reshape((-1,) + (1,) * theta.ndim)
    return theta.detach().unsqueeze(0) + scale * noise.detach()


def select_experts(scores: Tensor, k: int) -> Tensor:
    """Return long [k] indices of highest finite selection-set scores [N].

    1<=k<=N. Sort descending; ties retain lower original index first. Test labels
    are never inputs. This stable tie convention is specific to this exercise.
    """
    return torch.argsort(scores.detach(), descending=False, stable=True)[:k]


def majority_vote(predictions: Tensor, selected: Tensor, num_classes: int) -> Tensor:
    """Vote over selected models' hard predictions, returning long [B].

    predictions is long [N,B] with IDs in [0,num_classes), selected is a nonempty
    long [K] of distinct valid model indices. num_classes>=1, B may be zero.
    Count each selected expert equally, separately for each sample. Return the
    smallest class ID on tied counts. Never average weights, logits or class IDs.
    """
    answers = predictions[selected]
    counts = torch.stack([(answers == c).sum(dim=0) for c in range(num_classes)], dim=-1)
    return counts.argmax(dim=-1)

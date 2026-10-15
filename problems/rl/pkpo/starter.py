import math
import torch
from torch import Tensor


def _check(rewards, k):
    """Returns:
        Returns None after validating floating rewards [..., N], N>0, and integer 1 <= k <=
        N. Invalid inputs raise ValueError; it does not transform or return rewards.
    """
    if rewards.ndim < 1 or rewards.shape[-1] == 0 or not rewards.is_floating_point():
        raise ValueError("rewards must be floating [..., N] with N > 0")
    if type(k) is not int or not 1 <= k <= rewards.shape[-1]:
        raise ValueError("k must be an integer in [1, N]")


def _comb(n, k):
    """Returns:
        Returns a Python int: the binomial coefficient for 0 <= k <= n, otherwise 0. This
        supplied arithmetic helper does not return a Tensor.
    """
    return math.comb(n, k) if 0 <= k <= n else 0


def maxk_estimate(rewards: Tensor, k: int) -> Tensor:
    """U-statistic for expected max reward among k samples, along the last axis.

    Returns:
        Return one floating Tensor estimate with shape rewards.shape[:-1]: remove only the
        sample axis N from rewards [..., N]. Each value is the expected maximum over
        uniformly chosen size-k subsets. [N] input returns a scalar Tensor (), while [B, N]
        returns [B]. Preserve reward dtype/device and do not mutate inputs.
    """
    _check(rewards, k)
    raise NotImplementedError("Implement maxk_estimate")


def pkpo_weights(rewards: Tensor, k: int) -> Tensor:
    """Eq.33 sloo-minus-one coefficients, in original sample order, detached.

    Returns:
        Return one detached floating Tensor weights, with exactly the same [..., N] shape,
        dtype, device and original sample order as rewards. These are Eq. 33 coefficients,
        already including sample normalization; k=1 returns rewards / N. Keep the sample
        axis even when N=1.
    """
    _check(rewards, k)
    raise NotImplementedError("Implement pkpo_weights")


def pkpo_loss(log_probs: Tensor, rewards: Tensor, k: int) -> Tensor:
    """Negative policy gradient surrogate; weights already include sample normalization.

    Returns:
        Return one scalar floating Tensor loss, shape (): sum the negative weighted
        log-probabilities along the sample axis N, then average the remaining leading
        groups. Weights already include sample normalization; do not divide by N again. Keep
        gradients to log_probs, with no gradients to rewards/weights.
    """
    if log_probs.shape != rewards.shape:
        raise ValueError("log_probs and rewards must have matching shapes")
    raise NotImplementedError("Implement pkpo_loss")

"""EWC with an already supplied Fisher diagonal; no Fisher estimation is required."""
import torch
from torch import Tensor


def ewc_penalty(theta: Tensor, anchor: Tensor, fisher_diag: Tensor, strength: float) -> Tensor:
    """Return strength/2 times the SUM of Fisher-weighted squared parameter changes.

    Inputs are same-shaped finite CPU float32/float64 tensors with matching dtype.
    They may be scalar, empty or noncontiguous; fisher_diag is nonnegative.
    strength is a finite nonnegative Python scalar. Input validation is not required.
    Return a scalar Tensor in theta's dtype/device. Keep gradients only to theta;
    anchor and fisher_diag are fixed snapshots even when requires_grad=True.
    Zero strength, zero Fisher weights and empty inputs contribute differentiable
    zero. Do not mutate inputs. The caller adds the NEW task's loss separately.
    """
    raise NotImplementedError("Implement ewc_penalty")

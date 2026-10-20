# Bug: Add an EWC-style half absent from Eq. (4).
import torch
from torch import Tensor


def apply_prompts(images: Tensor, specific: Tensor, agnostic: Tensor) -> Tensor:
    """Add both [C,H,W] prompts to every image [B,C,H,W].

    Prompts are already placed on a full image canvas (zero outside their region).
    Return [B,C,H,W], retain all three gradients, preserve dtype, do not mutate or
    clip pixels. B may be zero; C,H,W are positive and all inputs match.
    """
    return images + specific.unsqueeze(0) + agnostic.unsqueeze(0)


def homeostatic_factor(importance: Tensor, displacement: Tensor,
                       xi: float = .01) -> Tensor:
    """Sum eta/(delta**2+xi) across completed domains, axis zero.

    Inputs are matching [T,*S] floating records, importance>=0, xi>0. Return
    detached S, including zeros for T=0. Records and output share dtype/device.
    Domain-boundary detection and path-importance estimation are supplied outside
    this exercise. Do not average domains, square their sum, or mutate inputs.
    """
    return (importance.detach() / (displacement.detach().square() + xi)).sum(dim=0)


def homeostatic_penalty(prompt: Tensor, anchor: Tensor, factor: Tensor,
                        strength: float) -> Tensor:
    """Return strength*sum(factor*(prompt-anchor)**2), with NO factor 1/2.

    Matching nonempty shape S; factor>=0, strength>=0. Only prompt is trainable;
    detach anchor and factor even if supplied with requires_grad=True. Return a
    scalar of the same dtype/device, without mutation. Inputs and intermediates
    are finite; this is the domain-agnostic prompt penalty, not the full loss.
    """
    return .5 * strength * (factor.detach() * (prompt - anchor.detach()).square()).sum()

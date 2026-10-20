# Contract: finite CPU float32/64 with matching dtype/device; preserve both.
# Support noncontiguous inputs; never mutate inputs. See the statement for full conventions.
import torch
from torch import Tensor


def apply_prompts(images: Tensor, specific: Tensor, agnostic: Tensor) -> Tensor:
    """images:[B,C,H,W], two canvases:[C,H,W] -> [B,C,H,W], B>=0.
    Add both already placed canvases without clipping; retain all gradients."""
    raise NotImplementedError()


def homeostatic_factor(importance: Tensor, displacement: Tensor,
                       xi: float = .01) -> Tensor:
    """Matching [T,*S] -> detached S: sum_T(importance/(displacement**2+xi)).
    T>=0; nonempty S; importance>=0, xi>0. Empty history -> zeros."""
    raise NotImplementedError()


def homeostatic_penalty(prompt: Tensor, anchor: Tensor, factor: Tensor,
                        strength: float) -> Tensor:
    """Matching nonempty S -> scalar strength*sum(factor*(prompt-anchor)**2).
    No 1/2. factor>=0, strength>=0. Detach anchor/factor, keep prompt gradients.
    Zero strength/factor -> differentiable zero."""
    raise NotImplementedError()

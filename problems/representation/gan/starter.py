from __future__ import annotations
import torch


def discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """
    Implement the discriminator's two independently averaged expectations and the stable differentiable non-saturating generator objective.

    Both functions consume finite float32/float64 logits and return a scalar with the same dtype/device, preserving every logit gradient. Inputs may have any shape, including scalars, but must be nonempty. Real/fake discriminator batches may have different shapes/counts and must share dtype/device. **Do not divide the sum by two** or smooth labels. Empty inputs or invalid dtypes raise ValueError. Do not modify inputs.

    The specified `L_G=-mean(log D(fake))` is non-saturating. The original minimax `mean(log(1-D(fake)))` is a legitimate different objective, but does not satisfy this contract. Never detach logits inside the losses. The caller detaches generated samples before D during its update; during G updates, freeze D parameters while preserving the input-to-G gradient.
    """
    raise NotImplementedError

def generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """
    Implement the discriminator's two independently averaged expectations and the stable differentiable non-saturating generator objective.

    Both functions consume finite float32/float64 logits and return a scalar with the same dtype/device, preserving every logit gradient. Inputs may have any shape, including scalars, but must be nonempty. Real/fake discriminator batches may have different shapes/counts and must share dtype/device. **Do not divide the sum by two** or smooth labels. Empty inputs or invalid dtypes raise ValueError. Do not modify inputs.

    The specified `L_G=-mean(log D(fake))` is non-saturating. The original minimax `mean(log(1-D(fake)))` is a legitimate different objective, but does not satisfy this contract. Never detach logits inside the losses. The caller detaches generated samples before D during its update; during G updates, freeze D parameters while preserving the input-to-G gradient.
    """
    raise NotImplementedError

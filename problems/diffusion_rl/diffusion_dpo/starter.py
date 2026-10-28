"""Implement the functions below; the statements specify the complete contracts."""
import torch
from torch import Tensor

def paired_noising(winner: Tensor, loser: Tensor, noise: Tensor,
                   alpha_bar: Tensor) -> tuple[Tensor, Tensor]:
    raise NotImplementedError("Implement paired_noising")

def diffusion_dpo_loss(model_w: Tensor, model_l: Tensor, ref_w: Tensor,
                       ref_l: Tensor, noise: Tensor, beta: float) -> Tensor:
    raise NotImplementedError("Implement diffusion_dpo_loss")

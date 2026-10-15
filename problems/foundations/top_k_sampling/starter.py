import torch
from torch import Tensor


def top_k_probs(logits: Tensor, k: int, temperature: float = 1.) -> Tensor:
    raise NotImplementedError

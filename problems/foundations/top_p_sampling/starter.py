import torch
from torch import Tensor


def top_p_probs(logits: Tensor, p: float, temperature: float = 1.) -> Tensor:
    raise NotImplementedError

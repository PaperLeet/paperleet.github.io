import torch
from torch import Tensor


def next_token_loss(logits: Tensor, tokens: Tensor) -> Tensor:
    raise NotImplementedError

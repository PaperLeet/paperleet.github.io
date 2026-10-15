import torch
from torch import Tensor


def sft_loss(logits: Tensor, tokens: Tensor, response_mask: Tensor, attention_mask: Tensor) -> Tensor:
    raise NotImplementedError

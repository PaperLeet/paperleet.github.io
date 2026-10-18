# BUG: Permit a previous conversation to predict the next conversation start.
import torch
from torch import Tensor


def assistant_target_mask(role_ids: Tensor, conversation_ids: Tensor, attention_mask: Tensor) -> Tensor:
    result = torch.zeros_like(attention_mask)
    result[:, 1:] = (
        (role_ids[:, 1:] == 2)
        & attention_mask[:, 1:] & attention_mask[:, :-1]
    )
    return result

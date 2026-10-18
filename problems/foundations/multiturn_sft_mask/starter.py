import torch
from torch import Tensor


def assistant_target_mask(role_ids: Tensor, conversation_ids: Tensor, attention_mask: Tensor) -> Tensor:
    """Return a bool [B,T] mask at TARGET positions, without shifting the output.

    CPU int64 roles/conversation IDs and bool attention_mask have identical [B,T]
    shapes; either dimension may be zero. Valid roles: 0=system/header, 1=user,
    2=assistant content (including its EOS), 3=tool. Valid conversation IDs are
    nonnegative; padding metadata is arbitrary. For j>0, select an assistant
    target only when j and j-1 are both valid and share a conversation ID.
    Column zero is always false. No input mutation or gradient is required.
    """
    raise NotImplementedError

import math
import torch
from torch import Tensor

def flash2_forward(q: Tensor, k: Tensor, v: Tensor, query_block: int, key_block: int, causal: bool=False) -> tuple[Tensor, Tensor]:
    """Attention output and per-row logsumexp.

q,k:(N,D), v:(N,V), returning O:(N,V) and L:(N,). Scores include 1/sqrt(D). query_block and key_block are positive integers. Causal visibility is key_index<=query_index; each row sees itself. Normalize U by l only after streaming. Both outputs retain gradients."""
    raise NotImplementedError()

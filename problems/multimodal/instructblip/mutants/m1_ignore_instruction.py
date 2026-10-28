# BUG: Removes instruction influence
import math
import torch
from torch import Tensor
import torch.nn.functional as F


def _floating(*values):
    first = values[0]
    for value in values:
        if not value.is_floating_point() or value.dtype != first.dtype or value.device != first.device or not torch.isfinite(value).all():
            raise ValueError("floating inputs must be finite and share dtype/device")

def instruction_query_block(queries: Tensor, instruction: Tensor, visual: Tensor, instruction_mask: Tensor) -> Tensor:
    _floating(queries,instruction,visual)
    if queries.ndim!=2 or queries.numel()==0 or instruction.ndim!=2 or visual.ndim!=2 or visual.numel()==0 or instruction.shape[1]!=queries.shape[1] or visual.shape[1]!=queries.shape[1] or instruction_mask.shape!=(instruction.shape[0],) or instruction_mask.dtype!=torch.bool:
        raise ValueError("expected compatible (Q,D), (I,D), (N,D) features and boolean instruction mask")
    context=torch.cat((queries,instruction*0),0)
    valid=torch.cat((torch.ones(queries.shape[0],dtype=torch.bool,device=queries.device),instruction_mask),0)
    scores=queries@context.T/math.sqrt(queries.shape[1])
    updated=queries+torch.softmax(scores.masked_fill(~valid[None,:],float('-inf')),-1)@context
    return updated+torch.softmax(updated@visual.T/math.sqrt(queries.shape[1]),-1)@visual

# Reverses the sign of backward transfer.
import torch
from torch import Tensor


def continual_metrics(scores: Tensor, baseline: Tensor) -> dict[str, Tensor]:
    tasks = scores.shape[0]
    acc = scores[-1].mean()
    if tasks == 1:
        zero = scores.new_zeros(())
        return {"acc": acc, "bwt": zero, "fwt": zero, "forgetting": zero}
    bwt = torch.stack([scores[j, j] - scores[-1, j] for j in range(tasks - 1)]).mean()
    fwt = torch.stack([scores[j - 1, j] - baseline[j] for j in range(1, tasks)]).mean()
    forgetting = torch.stack([
        scores[j:, j].max() - scores[-1, j] for j in range(tasks - 1)
    ]).mean()
    return {"acc": acc, "bwt": bwt, "fwt": fwt, "forgetting": forgetting}

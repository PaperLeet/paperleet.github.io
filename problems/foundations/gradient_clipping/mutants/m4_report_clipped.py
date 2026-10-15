# BUG: 返回裁剪后范数。
import torch
from torch import Tensor

def clip_gradients(gradients: list[Tensor], max_norm: float, eps: float = 1e-6) -> tuple[list[Tensor], Tensor]:
    maximum = max((float(g.abs().max()) for g in gradients if g.numel()), default=0.)
    if maximum == 0.:
        return [g.clone() for g in gradients], torch.zeros((), dtype=gradients[0].dtype)
    squared = torch.zeros((), dtype=torch.float64)
    for gradient in gradients:
        squared = squared + (gradient.to(torch.float64) / maximum).square().sum()
    unit_norm = squared.sqrt()
    total_norm = unit_norm * maximum
    # Scale the denominator to keep epsilon and tiny/large norms usable.
    denominator_scale = max(maximum, eps)
    denominator = unit_norm * (maximum / denominator_scale) + eps / denominator_scale
    scale = ((max_norm / denominator_scale) / denominator).clamp(max=1.)
    return [(g.to(torch.float64) * scale).to(g.dtype) for g in gradients], (total_norm * scale).to(gradients[0].dtype)

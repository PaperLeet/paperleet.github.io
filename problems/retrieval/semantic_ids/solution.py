import torch
from torch import Tensor


@torch.no_grad()
def residual_quantize(latents: Tensor, codebooks: Tensor) -> tuple[Tensor, Tensor]:
    residual = latents.detach().clone()
    reconstruction = torch.zeros_like(residual)
    selected = []
    for book in codebooks:
        distances = (residual[:, None, :] - book[None, :, :]).square().sum(-1)
        codes = distances.argmin(-1)
        contribution = book[codes]
        selected.append(codes)
        reconstruction = reconstruction + contribution
        residual = residual - contribution
    return torch.stack(selected, dim=1), reconstruction


def resolve_collisions(codes: Tensor) -> Tensor:
    counts, suffixes = {}, []
    for row in codes.tolist():
        key = tuple(row)
        suffixes.append(counts.get(key, 0))
        counts[key] = counts.get(key, 0) + 1
    suffix = torch.tensor(suffixes, dtype=torch.long, device=codes.device).reshape(-1, 1)
    return torch.cat([codes, suffix], dim=1)

import torch
import torch.nn.functional as F


def discriminator_loss(real_logits, fake_logits):
    if real_logits.numel() == 0 or fake_logits.numel() == 0:
        raise ValueError("both batches must be nonempty")
    if real_logits.dtype not in (torch.float32, torch.float64) or fake_logits.dtype != real_logits.dtype or fake_logits.device != real_logits.device:
        raise ValueError("matching floating dtype/device required")
    return F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()


def generator_loss(fake_logits):
    if fake_logits.numel() == 0 or fake_logits.dtype not in (torch.float32, torch.float64):
        raise ValueError("nonempty floating logits required")
    return F.softplus(-fake_logits).mean()

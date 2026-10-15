import math
import torch
from torch import Tensor
import torch.nn.functional as F


def contrastive_logits(image_features: Tensor, text_features: Tensor, logit_scale: Tensor) -> Tensor:
    """Required: L2-normalize [B,D] feature rows (eps=1e-12), dot products times exp(logit_scale),
    capped at 100. logit_scale is a scalar Tensor in log space. Return [B,B]; keep gradients
    through both encoders and unclipped temperature.

    Returns:
        Return one floating Tensor of shape [B,B]: row i is image i and column j is text j; B is
        the paired batch size. Preserve feature dtype/device and both encoder gradients, plus
        the gradient of scalar logit_scale below its cap (zero above the cap). Return scores,
        not probabilities, a tuple or a reduced loss; leave inputs unchanged.
    """
    raise NotImplementedError()


def clip_loss(image_features: Tensor, text_features: Tensor, logit_scale: Tensor) -> Tensor:
    """Required: diagonal image/text pairs are positive; all other pairs negative. Average cross
    entropy over rows and over columns to give a scalar loss. Call contrastive_logits; do not
    detach features or temperature.

    Returns:
        Return one scalar floating Tensor, shape (), with the feature dtype/device. It is the
        average of image-to-text and text-to-image batch-mean cross-entropies. Keep both feature
        gradients and the unclipped temperature gradient; do not detach or convert to a Python
        float. Leave inputs unchanged.
    """
    raise NotImplementedError()

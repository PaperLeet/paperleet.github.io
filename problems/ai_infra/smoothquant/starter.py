import math
import torch
from torch import Tensor

def smooth_scales(x: Tensor, weight: Tensor, alpha: float=0.5, eps: float=1e-05) -> Tensor:
    """Detached positive per-input-channel calibration scales.

x:(N,D), weight:(D,O), alpha in [0,1], eps>0. Clamp weight-channel absolute maxima to eps, then clamp final scales to eps. Zero activation channels follow this formula. Calibration scales are detached; smooth_transform is differentiable in x, weight and scale. Inputs are unchanged."""
    raise NotImplementedError()

def smooth_transform(x: Tensor, weight: Tensor, scales: Tensor) -> tuple[Tensor, Tensor]:
    """Differentiable reciprocally rescaled activations and weights.

x:(N,D), weight:(D,O), alpha in [0,1], eps>0. Clamp weight-channel absolute maxima to eps, then clamp final scales to eps. Zero activation channels follow this formula. Calibration scales are detached; smooth_transform is differentiable in x, weight and scale. Inputs are unchanged."""
    raise NotImplementedError()

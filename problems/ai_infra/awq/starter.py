import math
import torch
from torch import Tensor

def group_quantize(weight: Tensor, bits: int, group_size: int) -> Tensor:
    """Detached dequantized weights under asymmetric groupwise quantization.

x:(N,D), weight:(O,D); D is divisible by group_size. bits is an integer 2..8, and alphas is nonempty within [0,1]. Groups contain consecutive input channels of each output row. Quantizer step=max((max-min)/(2^bits-1),1e-5), zero=clip(round(-min/step),0,2^bits-1), with ties-to-even rounding. Clamp raw scales to 1e-4, then normalize by sqrt(max*min). Break equal-error ties by supplied alpha order. Return (scales,dequantized_weight,alpha,float MSE); offline calibration is detached."""
    raise NotImplementedError()

def awq_search(x: Tensor, weight: Tensor, bits: int, group_size: int, alphas: list[float]) -> tuple[Tensor, Tensor, float, float]:
    """Best input-channel scales, dequantized weights, exponent and output MSE.

x:(N,D), weight:(O,D); D is divisible by group_size. bits is an integer 2..8, and alphas is nonempty within [0,1]. Groups contain consecutive input channels of each output row. Quantizer step=max((max-min)/(2^bits-1),1e-5), zero=clip(round(-min/step),0,2^bits-1), with ties-to-even rounding. Clamp raw scales to 1e-4, then normalize by sqrt(max*min). Break equal-error ties by supplied alpha order. Return (scales,dequantized_weight,alpha,float MSE); offline calibration is detached."""
    raise NotImplementedError()

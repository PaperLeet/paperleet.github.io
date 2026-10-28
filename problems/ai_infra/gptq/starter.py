import torch
from torch import Tensor

def gptq_quantize(weight: Tensor, upper: Tensor, scales: Tensor, qmin: int=-7, qmax: int=7) -> Tensor:
    """Dequantized weights after sequential second-order error compensation.

weight:(O,D), upper:(D,D), scales:(O,). upper has zeros below its positive diagonal and represents H^{-1}=U^T U. Use ties-to-even rounding and inclusive integer bounds. Quantize left to right and compensate the current and all later columns. Return dequantized floating weights without mutating inputs; offline PTQ has no gradients."""
    raise NotImplementedError()

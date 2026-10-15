import torch
from torch import Tensor
NF4 = (-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0)

def nf4_quantize(weight: Tensor, block_size: int=64) -> tuple[Tensor, Tensor]:
    """Unpacked uint8 code indices plus per-block absmax; last block may be short.

    Returns:
        Return `(codes, scales)` as a Python tuple of tensors. codes has weight's
        shape/device, dtype torch.uint8 and unpacked NF4 indices 0..15 (one index per weight
        element). scales is a floating Tensor `[ceil(N/block_size)]` in weight's
        dtype/device, where N=weight.numel(); entries are per-block absolute maxima over the
        flattened weight, including a short final block. Both outputs are detached from
        weight; zero blocks use code 7 and scale 0. Do not mutate weight.
    """
    raise NotImplementedError('Implement nf4_quantize')

def nf4_dequantize(codes: Tensor, scales: Tensor, block_size: int=64) -> Tensor:
    """Frozen block scales; output shape matches codes, dtype/device follow scales.

    Returns:
        Return one floating Tensor with exactly codes' shape and scales' dtype/device,
        reconstructing the quantized weight. Do not return uint8 indices or flattened
        weights. Treat scales as frozen, so the result has no gradient path to scales;
        integer codes are nondifferentiable. Inputs remain unchanged.
    """
    raise NotImplementedError('Implement nf4_dequantize')

def qlora_linear(x: Tensor, codes: Tensor, scales: Tensor, a: Tensor, b: Tensor, alpha: float=1.0, block_size: int=64, bias: Tensor | None=None) -> Tensor:
    """Dequantized frozen base plus a differentiable alpha/r low-rank adapter.

    Returns:
        Return one floating Tensor `[...,out_features]` for x `[...,in_features]` and codes
        `[out_features,in_features]`, retaining all leading dimensions. Use x's
        dtype/device. The result combines the reconstructed frozen base with the low-rank
        adapter and optional bias. Preserve gradients to x, a and b only; scales and
        optional bias stay frozen and codes are integer indices. Do not mutate inputs or
        return quantization state.
    """
    raise NotImplementedError('Implement qlora_linear')

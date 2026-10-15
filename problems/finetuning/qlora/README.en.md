# QLoRA NF4 Blocks and Trainable Adapters

Implement block NF4 quantization and train adapters through a frozen quantized base.

Paper: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314v1). Anchors: Section 3: NormalFloat4 and frozen quantized base with LoRA; Appendix E: asymmetric NF4 with exact zero; bitsandbytes 0.43.3 functional.py get_4bit_type: pinned 16-level NF4 codebook.

## Selected core

### Return requirements

#### `nf4_quantize` — Required

Return `(codes, scales)` as a Python tuple of tensors. codes has weight's shape/device, dtype torch.uint8 and unpacked NF4 indices 0..15 (one index per weight element). scales is a floating Tensor `[ceil(N/block_size)]` in weight's dtype/device, where N=weight.numel(); entries are per-block absolute maxima over the flattened weight, including a short final block. Both outputs are detached from weight; zero blocks use code 7 and scale 0. Do not mutate weight.

#### `nf4_dequantize` — Required

Return one floating Tensor with exactly codes' shape and scales' dtype/device, reconstructing the quantized weight. Do not return uint8 indices or flattened weights. Treat scales as frozen, so the result has no gradient path to scales; integer codes are nondifferentiable. Inputs remain unchanged.

#### `qlora_linear` — Required

Return one floating Tensor `[...,out_features]` for x `[...,in_features]` and codes `[out_features,in_features]`, retaining all leading dimensions. Use x's dtype/device. The result combines the reconstructed frozen base with the low-rank adapter and optional bias. Preserve gradients to x, a and b only; scales and optional bias stay frozen and codes are integer indices. Do not mutate inputs or return quantization state.

### Formula and conventions

Three CPU functions. Code indices are unpacked uint8 for inspectability; packed 4-bit storage, double quantization and paged optimizers are outside scope.

$$
W\approx s_b C_{q(W/s_b)},\qquad y=x\mathrm{dequant}(W)^\top+\frac\alpha r xA^\top B^\top
$$

- `nf4_quantize(weight: Tensor, block_size: int=64) -> tuple[Tensor, Tensor]`
- `nf4_dequantize(codes: Tensor, scales: Tensor, block_size: int=64) -> Tensor`
- `qlora_linear(x: Tensor, codes: Tensor, scales: Tensor, a: Tensor, b: Tensor, alpha: float=1.0, block_size: int=64, bias: Tensor | None=None) -> Tensor`

## Contract and experiment

Implement block quantization, dequantization and a frozen-base adapter forward. The **provided `NF4` tuple** is the pinned asymmetric 16-value codebook: exact zero is index 7, positive and negative nonzero levels are not mirror images. Flatten the weight in row-major order, split into blocks, divide by each block's absmax, and choose the nearest level. Ties choose the lower code index. A zero block has scale zero and all code indices 7. A final short block has its own scale.

Return uint8 **unpacked indices** with the original weight shape and a flat floating-point scale vector. Decoding repeats each scale for its block. Weight quantization is frozen; dequantization freezes scales. The forward accepts A[r,in], B[out,r] and alpha/r scaling, preserves input/adapter gradients and freezes optional bias. `block_size` defaults to 64; tests use smaller blocks for inspection. NF4 constants are supplied in the starter; they are not something to memorize.

The toy trains adapters on a frozen quantized map. Storage here consumes one byte per index, not half a byte. It intentionally omits nibble packing, double quantization, paged optimizers and CUDA kernels, so its metrics cannot establish the paper's full memory claims. See the [official QLoRA training repository](https://github.com/artidoro/qlora) for those larger-system components.

## Toy provenance

**PaperLeet teaching adaptation.** PaperLeet synthetic regression with unpacked NF4 indices; does not measure full QLoRA memory savings. The test harness calls your submitted functions and records the resulting metrics and curves. The Lab's static preview comes from the reference; your Run/Submit artifacts come from your code. The experiment and test scaffolding are supplied in `experiment.py` and `tests/`.

## Sources and verification

[Official quantization library, pinned 0.43.3 NF4 codebook](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.43.3/bitsandbytes/functional.py). All exercise implementations and synthetic data are written for PaperLeet. The paper link pins the version used for the formulas.

T0 checks contracts, T1 compares with the reference, T2 checks mathematical and gradient properties, and T3 runs the toy through your implementation. Implement the starter functions in the online editor, then Run or Submit. The official large-model training system is not required.

# LoRA Low-rank Updates and Frozen-base Forward

Implement a scaled low-rank update while freezing base parameters.

Paper: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685v2). Anchors: Section 4.1, Eq. (3): W0 + BA and alpha/r scaling; Section 4.1: random A, zero B initialization and merged inference.

## Selected core

### Return requirements

#### `lora_delta` — Required

Return one floating Tensor `[out_features,in_features]`, the scaled low-rank weight update, from a `[r,in_features]` and b `[out_features,r]`. Preserve their shared dtype/device and gradients to both adapters. Return the update matrix only, without merging it into or modifying a base weight.

#### `lora_linear` — Required

Return one floating Tensor `[...,out_features]` from x `[...,in_features]`, preserving every leading batch/token dimension. It contains the base linear output plus adapter contribution, with optional bias included once. Preserve x's dtype/device and gradients to x, a and b; weight and optional bias are frozen even if they require gradients. Do not mutate any input or return updated parameters.

### Formula and conventions

Two functions for a linear layer; initialization and a rank-two regression training loop are provided.

$$
y=xW_0^\top+\frac\alpha r xA^\top B^\top
$$

- `lora_delta(a: Tensor, b: Tensor, alpha: float=1.0) -> Tensor`
- `lora_linear(x: Tensor, weight: Tensor, a: Tensor, b: Tensor, alpha: float=1.0, bias: Tensor | None=None) -> Tensor`

## Contract and experiment

Implement `lora_delta(A,B,alpha)` and `lora_linear(x,W,A,B,alpha,bias)`. Shapes are `A[r,in]`, `B[out,r]`, `W[out,in]`, and `x[...,in]`. The delta is `(alpha/r) * B @ A`. The forward operation is the frozen base linear map plus this low-rank update. Base weight and optional bias must not receive gradients, even when their inputs have `requires_grad=True`; gradients to `x`, `A`, and `B` must remain intact.

The training scaffold initializes A randomly and B to zero. Initially, the adapter changes no outputs: B has a nonzero learning signal while A's first gradient is zero. Freezing the entire base computation with `no_grad()` would incorrectly remove part of the gradient to x. Merge by adding the returned delta to W; the exercise compares merged and unmerged outputs. There is no dropout in this module.

The toy freezes a 6-by-8 map and fits a known rank-two change using 28 adapter parameters. The platform supplies input data, optimizer and 300 CPU training steps. Passing means recovering this controlled target, not reproducing a full language-model benchmark.

## Toy provenance

**PaperLeet teaching adaptation.** PaperLeet rank-two synthetic regression, not a language-model benchmark. The test harness calls your submitted functions and records the resulting metrics and curves. The Lab's static preview comes from the reference; your Run/Submit artifacts come from your code. The experiment and test scaffolding are supplied in `experiment.py` and `tests/`.

## Sources and verification

[Official Microsoft LoRA implementation](https://github.com/microsoft/LoRA/blob/main/loralib/layers.py). All exercise implementations and synthetic data are written for PaperLeet. The paper link pins the version used for the formulas.

T0 checks contracts, T1 compares with the reference, T2 checks mathematical and gradient properties, and T3 runs the toy through your implementation. Implement the starter functions in the online editor, then Run or Submit. The official large-model training system is not required.

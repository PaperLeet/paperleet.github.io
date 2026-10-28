# BLIP-2 maximum-query image-text matching

## 1. Problem

Implement `max_query_similarity(queries: Tensor, texts: Tensor, temperature: Tensor) -> Tensor`.

Normalize Q-Former outputs, compute per-text maximum-query similarities, and route gradients through winning queries; omit Transformer and frozen large-model execution.

## 2. Core equations

$$
S_{ij}=\frac{1}{\tau}\max_{q}\hat z_{iq}^{T}\hat t_j.
$$

## 3. From equations to code

Build normalized similarities of shape [B,Q,N], then reduce Q. Each text selects its own winning query; averaging queries beforehand loses this slot structure. Away from ties, gradients flow only through the winning query and then through normalization. Finite-difference tests avoid nondifferentiable ties.

## 4. Conventions

queries is [B,Q,D], texts is [N,D], and temperature is a positive scalar. Normalize along D and return [B,N] by maximizing independently over Q for every image-text pair. At ties, use torch.max first-index gradients. This kernel does not implement the full Q-Former.

CPU float32/float64; no input mutation. Finite inputs only. Tests use scalar oracles, boundary cases, appropriate gradient checks, and a submitted-function experiment.

## 5. Interactive figure

The deterministic CPU experiment calls your submitted function. The chart displays measured optimization, retrieval, or grouping behavior; inspect its metrics alongside the trace. Synthetic experiments do not reproduce paper benchmark scores.

## 6. Run

```bash
paperleet test multimodal/blip_2 --impl problems/multimodal/blip_2/solution.py
paperleet mutate multimodal/blip_2
paperleet vizdata multimodal/blip_2
```

## 7. Extensions

Read the cited mechanism, then extend the isolated kernel to the paper’s complete architecture and data pipeline. The exercise fixes a narrow, testable contract; its teaching simplifications are stated above.

Section 3.1: image-text contrastive learning takes the maximum similarity over query outputs

- [https://arxiv.org/abs/2301.12597](https://arxiv.org/abs/2301.12597)

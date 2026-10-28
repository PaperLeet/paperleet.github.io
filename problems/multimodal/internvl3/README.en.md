# InternVL3 variable visual position increments

## 1. Problem

Implement `variable_visual_positions(visual: Tensor, image_ids: Tensor, deltas: Tensor) -> Tensor`.

Per-image position increments, cumulative multimodal offsets, and gradients with an explicit origin of -1; exclude full rotary embeddings and InternVL training.

## 2. Core equations

$$
p_i=-1+\sum_{j=0}^{i}\left[(1-v_j)+v_j\delta_{\operatorname{image}(j)}\right].
$$

## 3. From equations to code

The recurrence depends on the current token’s modality and equals a prefix sum of increments minus one. Use one delta per image and never reset at image or text boundaries. Delta gradients count how many subsequent prefixes include each visual token. Delta one recovers positions 0 through L-1; a first visual token starts at delta-1 under the explicit origin.

## 4. Conventions

visual is boolean [L]; image_ids is int64 [L], using an image ID in [0,M) for visual tokens and -1 for text. deltas is floating [M] with 0<delta<=1. All tokens of an image share its delta. Start at p_-1=-1 and add 1 for the current text token or its image delta for the current visual token. Return [L] with deltas dtype/device. Empty sequences and text-only M=0 are valid; preserve delta gradients.

CPU float32/float64; no input mutation. Finite inputs only. Tests use scalar oracles, boundary cases, appropriate gradient checks, and a submitted-function experiment.

## 5. Interactive figure

The deterministic CPU experiment calls your submitted function. The chart displays measured optimization, retrieval, or grouping behavior; inspect its metrics alongside the trace. Synthetic experiments do not reproduce paper benchmark scores.

## 6. Run

```bash
paperleet test multimodal/internvl3 --impl problems/multimodal/internvl3/solution.py
paperleet mutate multimodal/internvl3
paperleet vizdata multimodal/internvl3
```

## 7. Extensions

Read the cited mechanism, then extend the isolated kernel to the paper’s complete architecture and data pipeline. The exercise fixes a narrow, testable contract; its teaching simplifications are stated above.

Section 2.1: Variable Visual Position Encoding, Equations 3 and 4

- [https://arxiv.org/abs/2504.10479](https://arxiv.org/abs/2504.10479)

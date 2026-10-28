# InstructBLIP instruction-aware query block

## 1. Problem

Implement `instruction_query_block(queries: Tensor, instruction: Tensor, visual: Tensor, instruction_mask: Tensor) -> Tensor`.

Two-stage query self-attention and visual cross-attention with identity projections, testing instruction influence and padding isolation; not a full Q-Former reproduction.

## 2. Core equations

$$
U=Q+\operatorname{softmax}\left(Q[Q;I]^T/\sqrt D+M\right)[Q;I],\quad O=U+\operatorname{softmax}(UV^T/\sqrt D)V.
$$

## 3. From equations to code

The first stage writes instruction information into queries; the second uses updated queries to retrieve visual features. Sending instructions only to a later language model bypasses this mechanism. Padding masks instruction keys while queries remain visible, ensuring every softmax row is defined. Both residuals belong to the declared reduced block; full projections, normalization, and FFNs remain extensions.

## 4. Conventions

queries is [Q,D], instruction [I,D], visual [N,D], and instruction_mask boolean [I]. First let queries attend to themselves and valid instruction tokens, with a residual. Then use updated queries for visual cross-attention, with another residual. Identity projections, one head, no LayerNorm/FFN define this reduced teaching block. All-masked instructions still leave query self-attention valid.

CPU float32/float64; no input mutation. Finite inputs only. Tests use scalar oracles, boundary cases, appropriate gradient checks, and a submitted-function experiment.

## 5. Interactive figure

The deterministic CPU experiment calls your submitted function. The chart displays measured optimization, retrieval, or grouping behavior; inspect its metrics alongside the trace. Synthetic experiments do not reproduce paper benchmark scores.

## 6. Run

```bash
paperleet test multimodal/instructblip --impl problems/multimodal/instructblip/solution.py
paperleet mutate multimodal/instructblip
paperleet vizdata multimodal/instructblip
```

## 7. Extensions

Read the cited mechanism, then extend the isolated kernel to the paper’s complete architecture and data pipeline. The exercise fixes a narrow, testable contract; its teaching simplifications are stated above.

Section 3.2 and Figure 2: instruction-aware visual feature extraction with query-text self-attention

- [https://arxiv.org/abs/2305.06500](https://arxiv.org/abs/2305.06500)

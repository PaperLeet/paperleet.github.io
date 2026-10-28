# Flamingo media-causal gated attention

## 1. Problem

Implement `gated_media_attention(x: Tensor, keys: Tensor, values: Tensor, text_media: Tensor, visual_media: Tensor, gate: Tensor) -> Tensor`.

Immediately preceding media masks, safe softmax for rows without media, and zero-initialized tanh gating in single-head identity-projection cross-attention.

## 2. Core equations

$$
y_i=x_i+\tanh(g)\sum_{j:m_j=t_i>0}\operatorname{softmax}_{j}(x_i^Tk_j/\sqrt D)v_j.
$$

## 3. From equations to code

Media IDs identify the immediately preceding medium, requiring equality rather than a less-than relation. Avoid softmax on all-negative-infinity rows by replacing them with safe rows and masking the resulting weights to zero. A tanh gate starts at an identity text path while retaining a nonzero derivative that can learn visual contributions.

## 4. Conventions

x is [T,D], keys/values are [N,D], and text_media/visual_media are int64 [T]/[N]. Media IDs start at 1; each text ID names its immediately preceding medium, with 0 for no medium. Only matching IDs are visible; rows without visible media return x exactly. gate is scalar and scales the residual through tanh. Use one head and identity projections, without Perceiver or FFN.

CPU float32/float64; no input mutation. Finite inputs only. Tests use scalar oracles, boundary cases, appropriate gradient checks, and a submitted-function experiment.

## 5. Interactive figure

The deterministic CPU experiment calls your submitted function. The chart displays measured optimization, retrieval, or grouping behavior; inspect its metrics alongside the trace. Synthetic experiments do not reproduce paper benchmark scores.

## 6. Run

```bash
paperleet test multimodal/flamingo --impl problems/multimodal/flamingo/solution.py
paperleet mutate multimodal/flamingo
paperleet vizdata multimodal/flamingo
```

## 7. Extensions

Read the cited mechanism, then extend the isolated kernel to the paper’s complete architecture and data pipeline. The exercise fixes a narrow, testable contract; its teaching simplifications are stated above.

Section 2.2 and Appendix A: single-image causal masking and tanh-gated cross-attention

- [https://arxiv.org/abs/2204.14198](https://arxiv.org/abs/2204.14198)

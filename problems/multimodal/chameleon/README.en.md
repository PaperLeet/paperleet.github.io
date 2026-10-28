# Chameleon per-head query-key LayerNorm

## 1. Problem

Implement `qk_layer_norm(x: Tensor, weight: Tensor, bias: Tensor, eps: float) -> Tensor`.

Per-head LayerNorm statistics, population variance, and affine parameters, with an attention-entropy experiment under growing input norms.

## 2. Core equations

$$
\mu_{th}=D^{-1}\sum_d x_{thd},\quad \sigma^2_{th}=D^{-1}\sum_d(x_{thd}-\mu_{th})^2,\quad y_{thd}=\gamma_{hd}\frac{x_{thd}-\mu_{th}}{\sqrt{\sigma^2_{th}+\epsilon}}+\beta_{hd}.
$$

## 3. From equations to code

Compute statistics independently for each token and head, never across tokens or heads. RMSNorm cannot replace mean-centering here. Use population variance with denominator D, divide by sqrt(var+eps), then apply per-head affine parameters. A constant head therefore outputs bias. Normalizing growing query/key norms keeps attention from rapidly saturating.

## 4. Conventions

x is [T,H,D]; weight/bias are [H,D]. Compute mean and population variance only over the last dimension D for each token/head; add positive eps inside the square root. Affine parameters are independent per head. Preserve all input gradients. Apply this kernel separately to queries and keys; RoPE and full multimodal generation are excluded.

CPU float32/float64; no input mutation. Finite inputs only. Tests use scalar oracles, boundary cases, appropriate gradient checks, and a submitted-function experiment.

## 5. Interactive figure

The deterministic CPU experiment calls your submitted function. The chart displays measured optimization, retrieval, or grouping behavior; inspect its metrics alongside the trace. Synthetic experiments do not reproduce paper benchmark scores.

## 6. Run

```bash
paperleet test multimodal/chameleon --impl problems/multimodal/chameleon/solution.py
paperleet mutate multimodal/chameleon
paperleet vizdata multimodal/chameleon
```

## 7. Extensions

Read the cited mechanism, then extend the isolated kernel to the paper’s complete architecture and data pipeline. The exercise fixes a narrow, testable contract; its teaching simplifications are stated above.

Section 2.3 Stability; ChameleonLayerNorm computes statistics only over head_dim

- [https://arxiv.org/abs/2405.09818](https://arxiv.org/abs/2405.09818)
- [https://github.com/huggingface/transformers/blob/main/src/transformers/models/chameleon/modeling_chameleon.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/chameleon/modeling_chameleon.py)

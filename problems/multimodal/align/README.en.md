# ALIGN bidirectional temperature contrast

## 1. Problem

Implement `align_loss(images: Tensor, texts: Tensor, temperature: Tensor) -> Tensor`.

Normalized dual-encoder features, differentiable temperature, and bidirectional batch softmax; train synthetic image-text pairs without web-scale noisy-data collection.

## 2. Core equations

$$
L=-\frac{1}{2B}\sum_i\left[\log\operatorname{softmax}_{j}(S_{ij})_i+\log\operatorname{softmax}_{j}(S_{ji})_i\right],\quad S_{ij}=\frac{\hat x_i^T\hat y_j}{\tau}.
$$

## 3. From equations to code

Normalize individual examples before constructing cross-batch similarities. Row softmax retrieves text from images; column softmax retrieves images from text, with different denominators. Select diagonal positives and average both directions. Temperature divides similarities and must remain a tensor in the computation graph.

## 4. Conventions

images and texts are nonempty floating [B,D] tensors; temperature is a positive scalar tensor. Matching indices are positive pairs. Normalize rows with norm floor 1e-12, keeping zero rows zero. Average over the batch in each retrieval direction, then average the two directions. Preserve all three gradients.

CPU float32/float64; no input mutation. Finite inputs only. Tests use scalar oracles, boundary cases, appropriate gradient checks, and a submitted-function experiment.

## 5. Interactive figure

The deterministic CPU experiment calls your submitted function. The chart displays measured optimization, retrieval, or grouping behavior; inspect its metrics alongside the trace. Synthetic experiments do not reproduce paper benchmark scores.

## 6. Run

```bash
paperleet test multimodal/align --impl problems/multimodal/align/solution.py
paperleet mutate multimodal/align
paperleet vizdata multimodal/align
```

## 7. Extensions

Read the cited mechanism, then extend the isolated kernel to the paper’s complete architecture and data pipeline. The exercise fixes a narrow, testable contract; its teaching simplifications are stated above.

Section 3: normalized embeddings and symmetric image-text contrastive loss

- [https://arxiv.org/abs/2102.05918](https://arxiv.org/abs/2102.05918)

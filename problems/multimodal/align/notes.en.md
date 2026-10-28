# ALIGN bidirectional temperature contrast — Solution

Normalize individual examples before constructing cross-batch similarities. Row softmax retrieves text from images; column softmax retrieves images from text, with different denominators. Select diagonal positives and average both directions. Temperature divides similarities and must remain a tensor in the computation graph.

$$
L=-\frac{1}{2B}\sum_i\left[\log\operatorname{softmax}_{j}(S_{ij})_i+\log\operatorname{softmax}_{j}(S_{ji})_i\right],\quad S_{ij}=\frac{\hat x_i^T\hat y_j}{\tau}.
$$

The reference implementation follows the declared layout and reduction:

```python
def align_loss(images: Tensor, texts: Tensor, temperature: Tensor) -> Tensor:
    _floating(images, texts, temperature)
    if images.ndim != 2 or images.numel() == 0 or texts.shape != images.shape or temperature.ndim != 0 or temperature.item() <= 0:
        raise ValueError("expected matching nonempty (B,D) features and positive scalar temperature")
    x = F.normalize(images, dim=-1, eps=1e-12)
    y = F.normalize(texts, dim=-1, eps=1e-12)
    scores = x @ y.T / temperature
    return -(F.log_softmax(scores, dim=1).diag().mean() + F.log_softmax(scores, dim=0).diag().mean()) / 2
```

Common errors caught by mutation tests:

- Drops text-to-image contrast
- Omits image normalization

Validation: T0 checks the contract; T1 uses an independent oracle; T2 covers boundaries, invariants, and gradients where defined; T3 executes the supplied function on deterministic synthetic inputs. See experiment.py for the complete inputs, update rule, step count and metrics. The gate thresholds are asserted in the T3 test.

Normalized dual-encoder features, differentiable temperature, and bidirectional batch softmax; train synthetic image-text pairs without web-scale noisy-data collection.

- [https://arxiv.org/abs/2102.05918](https://arxiv.org/abs/2102.05918)

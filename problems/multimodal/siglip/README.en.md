# SigLIP: Pairwise Image-Text Alignment

## 1. Task

A batch contains $B$ paired image and text features. Image $i$ matches text $i$; every other combination is negative. Implement the training core of [SigLIP](https://arxiv.org/abs/2303.15343): calibrated cosine logits and a stable pairwise sigmoid objective.

Validation and a small training scaffold are supplied. You implement `pairwise_logits` and `siglip_loss`.

## 2. Interfaces and return values

```python
pairwise_logits(images, texts, log_scale, bias) -> Tensor
siglip_loss(logits) -> Tensor
```

Both feature tensors have nonempty floating shape `(B,D)`. `log_scale` and `bias` are scalar tensors with the same dtype/device. The first function returns `(B,B)` logits; the second returns a scalar tensor of shape `()`.

Normalize each feature row independently:

$$
u_i=\frac{x_i}{\max(\lVert x_i\rVert_2,10^{-12})},\qquad
v_j=\frac{y_j}{\max(\lVert y_j\rVert_2,10^{-12})}.
$$

With log scale $s$ and bias $b$, compute

$$
\ell_{ij}=\exp(s)u_i^\top v_j+b.
$$

Use the additive bias convention from Algorithm 1 and the official implementation. Zero feature rows remain zero. All inputs and the exponential scale must be finite.

## 3. Objective and reduction

Let $z_{ii}=+1$ and $z_{ij}=-1$ otherwise. Compute

$$
L=\frac{1}{B}\sum_{i=1}^{B}\sum_{j=1}^{B}\operatorname{softplus}(-z_{ij}\ell_{ij}).
$$

Include every pair and divide the sum by **$B$**, rather than the number of pairs. No row or column softmax is involved. Use a stable softplus or log-sigmoid formulation so logits of magnitude 1000 produce finite losses and gradients.

For `zeros(4,4)`, the result is $4\log 2$. A batch of one still has one positive logistic term.

## 4. Gradients and validation

Preserve gradients into both feature towers, log scale and bias. Do not convert the calibration tensors to Python numbers, detach them, or modify any input.

Tests use scalar numerical oracles, analytic logit derivatives, finite differences for all four inputs, paired permutations and feature rescaling. The behavioral experiment performs 120 real CPU gradient updates on four synthetic pairs and the calibration parameters. It checks retrieval and separation of positive/negative logits; it does not train image or text encoders.

## 5. Frequent mistakes

- Averaging over all pairs instead of dividing their sum by the batch size.
- Removing negative pairs or using a batch softmax objective.
- Normalizing across examples instead of each feature vector.
- Reversing the bias sign or disconnecting the text features from autograd.

## 6. Run

```bash
paperleet init multimodal/siglip
paperleet run multimodal/siglip
```

## 7. Sources

[Paper](https://arxiv.org/abs/2303.15343), Section 3.2 and Algorithm 1; [official training code](https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py). This exercise uses a complete single-device pair matrix; distributed chunk exchange is outside its implementation scope.

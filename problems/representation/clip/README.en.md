# Symmetric Contrastive Learning

[CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) learns a shared space from matched images and text. Figure 3 gives the contrastive training objective; §2.5 describes temperature initialization and scale clipping. The author's released model implementation is [openai/CLIP](https://github.com/openai/CLIP).

## Your core module

### Return requirements

#### `contrastive_logits` (required)

Return one floating `Tensor` of shape `[B,B]`: row i is image i and column j is text j; B is the paired batch size. Preserve feature dtype/device and both encoder gradients, plus the gradient of scalar `logit_scale` below its cap (zero above the cap). Return scores, not probabilities, a tuple or a reduced loss; leave inputs unchanged.

#### `clip_loss` (required)

Return one scalar floating `Tensor`, shape `()`, with the feature dtype/device. It is the average of image-to-text and text-to-image batch-mean cross-entropies. Keep both feature gradients and the unclipped temperature gradient; do not detach or convert to a Python float. Leave inputs unchanged.

### Formula and conventions

Implement `contrastive_logits` and `clip_loss`. Image and text encoder outputs are supplied as `[B,D]` tensors. Normalize each row, form all pairwise similarities, and train both retrieval directions. Pair `i` matches pair `i`; the remaining examples in the batch act as negatives.

$$
\bar i=i/\|i\|_2,\quad \bar t=t/\|t\|_2,\quad
S=\min(e^s,100)\,\bar i\bar t^\top,
$$
$$
L=\tfrac12\big[\operatorname{CE}(S,\operatorname{arange}(B))+
\operatorname{CE}(S^\top,\operatorname{arange}(B))\big].
$$

`s` is a scalar **log inverse temperature**, not the temperature itself. This exercise enforces the paper's cap of 100 in the forward function by clipping in log space before exponentiating. This remains finite even when a supplied `s` is extremely large. Below the cap, preserve its gradient; above the cap, its gradient is zero.

| Item | Convention |
|---|---|
| Features | Same batch size and width, floating tensors `[B,D]`; encoders already ran |
| Normalization | Rowwise L2 norm with epsilon `1e-12`; a zero row stays zero |
| Positives | Diagonal matched pairs; batch is nonempty |
| `contrastive_logits` | `[B,B]`, rows images and columns text |
| `clip_loss` | Scalar mean of both directional mean cross-entropies |
| Gradients | Both feature tensors and the unclipped scalar `logit_scale` |

The released repository primarily supports pretrained-model inference; this exercise specifies the paper's training loss and clipping behavior explicitly. No image or text encoder implementation is required.

## Example

For two matched orthogonal unit vectors and `s=log(2)`, logits equal `2I`. Both directions have loss `log(1+exp(-2))`, approximately 0.1269. A nonsymmetric similarity matrix generally has different image-to-text and text-to-image losses, so testing only identical features would miss a one-direction bug.

## Toy experiment

The provided experiment optimizes eight pairs of random 8-dimensional vectors and a learnable temperature for 80 steps. It records loss, cosine similarities and retrieval accuracy in both directions. The experiment calls the submitted loss, while retrieval is independently evaluated from the learned normalized features.

**Provenance:** PaperLeet teaching adaptation of Figure 3, using trainable vectors instead of large image/text encoders. It does not reproduce CLIP pretraining or zero-shot benchmarks. No datasets or weights are downloaded. T3 requires both retrieval accuracies to reach 1 and final loss below .05.

## Validation

T0 checks one pair and zero-feature stability; T1 derives the orthogonal example; T2 verifies the independently derived symmetric loss, feature-scale invariance, pair permutation, temperature behavior and all gradients. T3 trains paired retrieval. Known incorrect variants omit normalization, remove one direction, misuse the log scale or omit its cap.

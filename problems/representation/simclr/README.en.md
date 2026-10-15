# SimCLR: NT-Xent and same-view negatives

Implement `nt_xent_loss(view1, view2, temperature=.5)` for projected vectors. The objective follows [SimCLR](https://proceedings.mlr.press/v119/chen20j.html), Section 2.1 / Equation 1 and Algorithm 1.

## Exercise contract

Both finite floating inputs have identical nonempty `[B,D]` shapes and a shared float32/float64 dtype and device. Return a scalar preserving dtype/device, leave inputs intact, accept strided views, and preserve derivatives into both inputs. The supplied validation rejects empty axes, mismatched shapes/dtypes, NaN/Inf and temperatures other than finite positive Python int/float values (excluding bool).

Stack **all of view1, then all of view2**. The partner of anchor `i` is `i+B` for `i<B`, otherwise `i-B`. The paper's interleaved indexing must be translated to this exercise's stacked layout.

$$
z_i=\frac{v_i}{\max(\lVert v_i\rVert_2,10^{-12})},\quad
s_{ij}=z_i^\top z_j/\tau,\quad
L=\frac{1}{2B}\sum_i\left[\operatorname{logsumexp}_{j\ne i}s_{ij}-s_{i,p(i)}\right].
$$

Exclude only the anchor itself: each row has one positive and `2B-2` negatives, including other instances in the **same view**. Keep the positive in the denominator, average both directions, and use a stable logsumexp or cross entropy. The epsilon and zero-vector behavior are explicit exercise conventions.

For two identical orthogonal pairs, loss is `log(1+2*exp(-1/temperature))`. A single pair has loss and both gradients equal to zero. Identical nonzero vectors collapse to loss `log(2B-1)`.

## Compare with CLIP

The CLIP exercise compares each image against text candidates. Reusing only its `[B,B]` cross-view objective here drops same-view negatives. This exercise also uses a fixed supplied temperature, rather than learned logit scale.

Independent scalar loops, finite differences for both inputs, paired permutations, view swapping, a shared orthogonal basis change, positive row rescaling, zero/near-zero rows and extreme temperatures check the implementation. Eight mutants cover normalization, self masking, candidate membership, partner indices, direction, reduction scale, temperature and detached gradients.

## Actual CPU training lab

The supplied original fixture trains two independently initialized `6x8` tables: 96 scalars, 120 full-batch Adam updates, learning rate `.07`, temperature `.2`. Every ten updates, record actual loss, average positive/hardest-negative cosine and training paired-retrieval accuracy. A final bar chart compares each anchor's positive to its hardest negative. A strict top-1 hit requires the positive to beat all other non-self candidates; ties are incorrect. Independent scalar measurements cross-check the objective.

This is a **PaperLeet synthetic teaching adaptation**. It learns vectors directly, without image augmentation, an encoder, a projection head, real datasets or downstream evaluation. Training metrics are not paper benchmark results. CPU/WASM needs no GPU, account, weights or downloads; unsupported custom PyTorch operations retain the native judge path.

## Sources and reduction scale

The [paper's Algorithm 1](https://proceedings.mlr.press/v119/chen20j/chen20j.pdf) averages over `2B` anchors. The [authors' TensorFlow objective](https://github.com/google-research/simclr/blob/master/objective.py) adds its separately reduced directional losses, yielding twice this exercise's mean under the default reduction. Align scales before comparing logs or learning rates. The validation contract, synthetic fixture and tests are original exercise choices.

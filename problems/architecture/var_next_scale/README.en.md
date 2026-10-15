# VAR Next-Scale Prediction

[VAR](https://arxiv.org/html/2404.02905v2) predicts a whole resolution level at a time. Implement two functions that prepare frozen tokenizer features and the training attention mask. The [official implementation](https://github.com/FoundationVision/VAR/blob/main/models/quant.py) constructs each input from a cumulative reconstruction of **earlier** token maps.

## Implement

### Return requirements

#### `scale_causal_mask` — Required

Return one `torch.bool` Tensor `[L,L]` on the requested device, with L=sum(n*n for n in scales). Rows are query tokens and columns key tokens, concatenated by increasing scale and raster order within each scale. True allows the same or an earlier scale, including every within-scale pair. It is a visibility matrix without gradients, not an additive floating-point mask.

#### `next_scale_inputs` — Required

Return one floating Tensor `[B,L_next,C]`: B batch size, C codebook channel width and L_next=sum(n*n for n in scales[1:]). Concatenate features for target scales 2..K in scale order, with row-major spatial tokens within each scale; do not include the separate first-scale start embedding. Preserve codebook dtype/device. A single scale returns `[B,0,C]`. The frozen-tokenizer result must have requires_grad=False even if codebook/projections require gradients; do not mutate inputs.

#### `lift_codes` — Provided

Return one floating Tensor `[B,C,F,F]`, where B is batch size, C the projected channel width and F=full_size. It contains codebook features lifted to full spatial resolution and channel-projected, in channels-first layout. Preserve the shared codebook/projection dtype/device and input values. This helper itself does not detach gradients from codebook/projection; next_scale_inputs calls it in a no-grad context.

#### `_validate_codes` — Provided helper

Return a Python `list[int]` of square side lengths, one per token map in input order, after validating the scale schedule and tensor layouts. It is not a Tensor. Raise ValueError for the checked invalid inputs; do not modify any input.

### Formula and conventions

`next_scale_inputs(token_maps, codebook, projections) -> Tensor`

The list contains `K` integer target maps `[B,n_k,n_k]`. Let `lift_codes` be the supplied lookup → bicubic upsample → channel projection helper, and let `H=n_K`. For the input predicting scale `k>0`, compute

$$F_{<k}=\sum_{j<k}\operatorname{lift\_codes}(r_j,Z,P_j,H),\qquad x_k=\operatorname{area}_{n_k}(F_{<k}).$$

Flatten each `x_k` in row-major spatial order into `[B,n_k²,C]`, then concatenate for `k=1,…,K−1`. The result is `[B,Σ_{k>0}n_k²,C]`. The first scale uses a separate class/start embedding and contributes no row to this function. For `K=1`, return `[B,0,C]`. The last target map is never used as an input.

`scale_causal_mask(scales, device=None) -> Tensor`

Return boolean `[L,L]`, with `L=Σn_k²` and `True` meaning **visible**. Query at scale `i` may read a key at scale `j` exactly when `j≤i`. Thus every same-scale pair is visible, even in reverse raster order. [The official mask](https://github.com/FoundationVision/VAR/blob/main/models/var.py) follows this rule. Same-scale attention is safe because those positions contain lower-scale features, not their target tokens.

## Contract

Scales are positive, strictly increasing integers beginning at `1`; all maps are square and share `B>0`. Token maps are int64 with indices in `[0,V)`. The shared codebook is `[V,C]` and the supplied projections are `[K,C,C]`. Inputs are finite CPU float32/float64 tensors with matching floating dtype/device; strided tensors are valid. Preserve values, dtype and device. The tokenizer is frozen: the supplied `no_grad` wrapper is intentional.

`lift_codes` is already implemented. It uses `align_corners=False` bicubic interpolation to the largest resolution, followed by `P_k @ feature` at each pixel. In the original system `Phi` contains learned spatial convolutions; fixed channel matrices keep this exercise focused. Position embeddings, class conditioning, codebook training and the Transformer are outside this contract.

## Toy

A teaching adaptation uses 1×1, 2×2 and 4×4 target maps and a fixed attention layer. Your two functions control the entire information path. Changing a target map must not change predictions at its own or any coarser scale; earlier maps must still influence finer predictions. The chart also compares visible-key counts with raster causal attention. This is an information-flow experiment, not an image-generation benchmark.

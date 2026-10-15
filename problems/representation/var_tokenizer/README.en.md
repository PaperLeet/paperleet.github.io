# VAR Multiscale Residual Quantization

Implement the frozen latent tokenizer behind [VAR](https://arxiv.org/html/2404.02905v2). Each scale quantizes what the earlier scales have left unexplained. Algorithms 1–2 define this encoding/reconstruction loop; [the official implementation](https://github.com/FoundationVision/VAR/blob/main/models/quant.py) gives the area and bicubic interpolation conventions.

## Implement

### Return requirements

#### `multiscale_quantize` (required)

Return `(token_maps, reconstructions)`, a Python tuple of two Python lists, each of length K=`len(scales)` in scale order. Entry k of `token_maps` is an int64 `Tensor` `[B,n_k,n_k]`; entry k of `reconstructions` is a detached floating `Tensor` `[B,C,H,H]` containing the cumulative reconstruction through scale k, not just its contribution. B is batch size, C channels and H the final scale. Preserve feature dtype/device; snapshots must have independent storage. No input mutation or gradients.

#### `decode_multiscale` (required)

Return one detached floating `Tensor` `[B,C,H,H]`, where B is token-map batch size, C codebook width and H the largest map size. It is the sum of every lifted scale, matching the encoder's final reconstruction, not a list or scale average. Preserve codebook dtype/device and leave all inputs unchanged; no gradients are required.

#### `lift_codes` (provided)

Provided helper: return one floating `Tensor` `[B,C,full_size,full_size]` from int64 tokens `[B,n,n]`, codebook `[V,C]` and channel projection `[C,C]`. Keep the codebook/projection dtype and device. The helper itself does not detach differentiable codebook/projection paths; the frozen encode/decode callers run under no_grad. No input mutation.

#### `resize_residual` (provided)

Provided helper: return a floating `Tensor` `[B,C,size,size]` from residual `[B,C,H,H]`, preserving dtype/device and its gradient path. If `size==H`, return the original tensor itself (an alias), not a copy or `None`. Otherwise return the resized values. Do not modify residual in place.

#### `_validate_codes` (provided)

Provided helper: return a Python `list[int]` containing the K square token-map sizes in input order, not tensors or `None`. Validate nonempty int64 maps `[B,n_k,n_k]`, increasing scales beginning at 1, in-range code indices, codebook `[V,C]` and matching projections `[K,C,C]`; checked failures raise `ValueError`. Leave inputs unchanged; there is no gradient output.

### Formula and conventions

`multiscale_quantize(features, codebook, scales, projections) -> (token_maps, reconstructions)`

Start with residual `e=features` and full-resolution reconstruction `f_hat=0`. For every scale `n_k`, in ascending order:

1. Call supplied `resize_residual(e,n_k)` and choose the nearest shared codebook vector at each spatial position, minimizing squared Euclidean distance. Break exact ties with the lowest code index.
2. Use supplied `lift_codes(indices,codebook,projections[k],H)` to turn the selected codes into a full-resolution contribution `h_k`.
3. Update `e=e−h_k` and `f_hat=f_hat+h_k`. Save the integer map and an independent snapshot of `f_hat`.

Return `K` int64 maps `[B,n_k,n_k]` and `K` cumulative floating reconstructions `[B,C,H,H]`. The last scale quantizes the remaining full-resolution residual; do not quantize the original feature map repeatedly.

`decode_multiscale(token_maps, codebook, projections) -> Tensor`

Lift every map to the largest resolution and sum the contributions. Return `[B,C,H,H]`. The same token maps and projections must recover the encoder's final cumulative reconstruction.

## Provided helpers and contract

`resize_residual` uses area interpolation; `lift_codes` performs lookup, bicubic upsampling with `align_corners=False`, then a channel matrix multiplication. Both are provided. The original learned spatial `Phi` layers are replaced by given `[K,C,C]` matrices in this teaching exercise. No cosine-normalized code selection is used.

`features` is finite, nonempty, square `[B,C,H,H]`. The shared codebook is `[V,C]`, with `V,C>0`. All floating inputs are CPU float32/float64 and share dtype/device; noncontiguous tensors are supported. Scales are strictly increasing positive integers starting at `1` and ending at `H`; `K=1,H=1` is valid. Input token maps to decoding obey the same scale schedule and use in-range int64 indices. Do not mutate inputs or alias the saved reconstruction snapshots.

This is a frozen tokenizer. The provided `no_grad` wrappers are intentional: no straight-through estimator, codebook update, commitment loss, encoder or image decoder is required.

## Toy

A teaching adaptation quantizes an 8×8 latent containing coarse regions and a small alternating detail pattern, using nine scalar codes and identity projections. Your encoding and decoding functions produce a reconstruction-error curve and a latent row plot. The example checks detail recovery and encode/decode agreement. It does not assume that each intermediate scale must always reduce error for arbitrary codebooks/projections, and does not reproduce ImageNet training.

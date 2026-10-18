# Block Diffusion Attention

## 1. Exercise goal

[Block Diffusion](https://arxiv.org/html/2503.09573v2), ICLR 2025 Oral, denoises a block while conditioning on earlier clean blocks. Its vectorized training pass concatenates **`[noisy tokens, clean tokens]`**. Implement the specialized attention mask and apply it to projected Q/K/V.

## 2. Interfaces and returns

### Return requirements

#### `block_diffusion_mask` (required)

Return one torch.bool Tensor of shape [2L, 2L], where L=length. Rows are queries and columns are keys; both axes concatenate L noisy tokens then L clean tokens. True means allowed. Use the requested device (CPU by default in this exercise); the mask has no gradients and is not an additive floating mask.

#### `block_diffusion_attention` (required)

Return the attention output Tensor [B, H, 2L, Dv], matching value: B is batch size, H heads, L tokens per stream, and Dv value width. Return aggregated features, not the [B,H,2L,2L] attention weights. Preserve input float32/float64 dtype, device and gradients through Q/K/V; do not mutate inputs.

### Equations and interface notes

For a sequence of length `L`, let `b(i)=floor(i/block_size)` be its local block index. Appendix B.6 defines the `2L × 2L` allowed-position mask:

$$
M=\begin{bmatrix}M_{BD}&M_{OBC}\\0&M_{BC}\end{bmatrix},\quad
M_{BD}[i,j]=[b(i)=b(j)],\quad M_{OBC}[i,j]=[b(j)<b(i)],\quad M_{BC}[i,j]=[b(j)\le b(i)].
$$

Rows are queries; columns are keys. A noisy query sees its **entire noisy block** and **strictly earlier clean blocks**. A clean query sees its own and earlier clean blocks, and never sees noisy tokens. Allowing the current clean block into a noisy query leaks its training target.

### Interfaces

- `block_diffusion_mask(length, block_size, device=None)` returns a boolean `[2L,2L]` tensor; `True` means allowed. Both sizes must be positive Python integers.
- `block_diffusion_attention(query, key, value, block_size)` uses Q/K `[B,H,2L,D]`, V `[B,H,2L,Dv]`, and returns V's shape. Compute scaled dot-product attention with `sqrt(D)`, set forbidden logits to `−∞`, then normalize over keys.

Use finite, nonempty CPU float32/float64 tensors, including strided views. Preserve dtype/device, gradients and input values. Validation is supplied. Every query has at least one allowed key. The final partial block is supported as a **teaching extension**; the paper's main notation assumes equal full blocks. A block size at least `L` creates one block.

This module covers training attention dependencies. It does not implement the discrete corruption process, NELBO, a sampler, KV caching or a GPU kernel.

## Toy: the original six-token example

**Paper-derived adaptation.** Appendix B.6 Figure 3 uses `L=6` and block size 2. We independently reconstruct that mask, then add tiny deterministic attention and perturb each value token in turn. The charts show the number of allowed keys and measured influence of two clean tokens. Forbidden influences must be zero; allowed ones must remain observable.

For example, noisy query index 2 can read concatenated keys `[2,3,6,7]`; clean query 6 reads `[6,7]`. The preview uses the reference, and T3 uses your submitted mask and attention. The numerical oracle evaluates each block's conditioning context separately. [Author code](https://github.com/kuleshov-group/bd3lms).

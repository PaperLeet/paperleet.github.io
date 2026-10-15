# Sinusoidal Positional Encoding

The Transformer has no recurrence or convolution to reveal token order. [Attention Is All You Need, §3.5](https://arxiv.org/html/1706.03762v7#S3.SS5) adds fixed sine/cosine features to token embeddings. Implement the positional encoding itself; embedding lookup, attention and training are separate exercises.

## Your core module

### Return requirements

#### `sinusoidal_encoding` — Required

Return one floating Tensor `[length,d_model]`, without a batch dimension. Row i encodes absolute position start+i; columns are feature channels, with sine in even channels and cosine in odd channels. Honor the requested float32/float64 dtype and device. Odd widths retain the final sine channel; length=0 returns `[0,d_model]`. These fixed encodings have requires_grad=False; return the Tensor rather than a list or a pair of tables.

### Formula and conventions

Implement `sinusoidal_encoding(length, d_model, start=0, *, dtype=torch.float32, device=None)`:

$$
PE_{p,2i}=\sin\left(p/10000^{2i/d_{\mathrm{model}}}\right),\qquad
PE_{p,2i+1}=\cos\left(p/10000^{2i/d_{\mathrm{model}}}\right).
$$

The sine and cosine in one pair share a frequency. Use the full model width in the exponent, not the number of pairs.

| Contract | Convention |
|---|---|
| Output | `[length, d_model]` for integer positions `start` through `start + length - 1` |
| Dimensions | Integer `length >= 0`, `d_model > 0`, `start >= 0`; invalid values raise `ValueError` |
| Precision | Explicit `torch.float32` or `torch.float64`; unsupported dtype raises `ValueError` |
| Device | Construct on the requested device; tests run on CPU |
| Odd width | Keep the final unpaired sine channel; its frequency still uses the full width |
| Empty sequence | Return an empty `[0, d_model]` tensor |
| Gradients | Fixed encoding has no trainable parameters; adding it to embeddings preserves their gradient |

Odd widths, an offset and explicit dtype are PaperLeet interface extensions. The original model uses even width 512. The model separately multiplies token embeddings by the square root of their width (§3.4); do not apply that scale to the positional encoding. No dropout is required here.

At position zero, a width-five encoding is `[0, 1, 0, 1, 0]`. For a batch of embeddings `x` shaped `[B,L,D]`, `x + sinusoidal_encoding(L,D,dtype=x.dtype,device=x.device)` broadcasts one set of positions across all batches.

## Toy experiment

A constant relative offset is a fixed rotation of each `[sin, cos]` pair:

$$
\sin(\theta+\delta)=\sin\theta\cos\delta+\cos\theta\sin\delta,\quad
\cos(\theta+\delta)=\cos\theta\cos\delta-\sin\theta\sin\delta.
$$

The provided experiment generates 32 positions with eight features, advances every position by seven and compares the submitted output to this independently calculated rotation. It also measures the closest pair of distinct position vectors. The lab plots actual and rotated features and reports their maximum error.

**Provenance:** a PaperLeet teaching adaptation of the relative-offset property discussed in §3.5. The paper does not report this 32-position experiment. T3 invokes your implementation; the reference preview is labeled separately.

## Validation

T0 checks shape, precision, device and empty output. T1 compares against an independent scalar formula, including large offsets and odd widths. T2 verifies unit sine/cosine pair norms, chunk offsets, embedding gradients and invalid arguments. T3 checks the relative-offset toy. Common bugs are halving the frequency exponent, using sine for both channels, ignoring offsets and discarding requested precision.

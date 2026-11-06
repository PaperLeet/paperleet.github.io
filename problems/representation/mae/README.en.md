# MAE: masked pixel reconstruction

## 1. Problem

[Masked Autoencoders](https://arxiv.org/abs/2111.06377) learn representations by predicting image content removed from the encoder's input. The original method uses an asymmetric encoder–decoder and a high masking ratio. This exercise isolates its pixel reconstruction objective on already patchified tensors.

Implement `normalize_patch_targets(target, eps=1e-6)` and `masked_reconstruction_loss(prediction, target, mask, norm_pix_loss=False, eps=1e-6)`. Prediction and target have shape `[B,N,P]`: images, patches, and flattened pixel coordinates. `mask[b,n]=True` means a removed patch whose reconstruction is supervised. Return normalized targets of the same shape and one scalar loss, respectively.

## 2. Core equations

Optional target normalization acts independently on each patch. To match the [released `forward_loss`](https://github.com/facebookresearch/mae/blob/main/models_mae.py), use **sample variance**: the original `target.var(dim=-1)` divides by `P-1`.

$$
\mu_{bn}=\frac1P\sum_p x_{bnp},\quad
v_{bn}=\frac1{P-1}\sum_p(x_{bnp}-\mu_{bn})^2,\quad
\hat x_{bnp}=\frac{x_{bnp}-\mu_{bn}}{\sqrt{v_{bn}+\epsilon}}.
$$

Let $t=x$ for raw targets and $t=\hat x$ for normalized targets. With $M=\sum_{bn}m_{bn}$:

$$
e_{bn}=\frac1P\sum_p(\mathrm{prediction}_{bnp}-\operatorname{sg}(t_{bnp}))^2,\qquad
\mathcal L=\frac{\sum_{bn}m_{bn}e_{bn}}{\max(M,1)}.
$$

`sg` treats the supplied target as fixed data inside the loss. The standalone normalizer remains differentiable. Detaching supplied targets and defining the empty-mask result as zero make the exercise boundary explicit; official MAE receives image data and trains with nonempty masks.

Example: `[-1,1]` has sample variance `2`, so normalization divides by `sqrt(2+eps)`. If one image has one masked patch and another has two, all three patches receive equal weight.

## 3. From equations to code

1. Center over the final pixel axis, preserving dimensions for broadcasting.
2. Sum squared centered pixels and divide by `P-1`; add epsilon inside the square root.
3. In the loss, detach the target and normalize it only when `norm_pix_loss=True`.
4. Average pixel squared error within each patch. Multiply by the mask and divide the total by the number of removed patches, clamped to at least one.

Keep the provided input checks. The loss may call this file's normalizer; neither function may import a reference implementation. Do not detach predictions or normalize predictions automatically.

## 4. Exercise conventions

- Nonempty `[B,N,P]` tensors share finite float32/float64 values, dtype, and device. Support strided inputs and preserve every input. Mask is Boolean `[B,N]` on the same device.
- Raw reconstruction allows `P=1`. Normalized targets require `P>=2`. Constant patches normalize to zero.
- `eps` is a positive finite Python int/float; reject bool and Tensor scalars. `norm_pix_loss` is a Python bool.
- Average over **all masked patches**, not over image-wise averages. Visible patches have exactly zero prediction gradient. An empty mask returns a scalar zero connected to predictions with zero gradient.
- A per-patch offset leaves normalized targets unchanged. For nonzero scale `s`, scaling targets by `s` and epsilon by `s²` multiplies normalized targets by `sign(s)`; epsilon remains positive.

Tests independently compute scalar values, derive prediction gradients, finite-difference the normalizer's VJP, and cover unequal masks, offsets, replication, empty/full masks and one-pixel boundaries.

## 5. Interactive charts and CPU experiment

The Lab trains a **shared 3-to-4 linear decoder**, including bias, with 16 trainable model parameters. There are 24 deterministic trigonometric context vectors arranged as six images of four patches; 14 patches are supervised. Targets come from one fixed linear map. Adam runs **120 actual updates at learning rate `.06`**, using your raw-target loss and recording every ten steps. Targets are included in the optimizer deliberately: correct gradient stopping keeps them unchanged.

Charts compare your objective with an independent Python masked-pixel MSE and show one reconstructed patch. On the reference CPU run, MSE fell from about `.671` to `6.6e-6`. T3 requires MSE below `1e-4` and below `.001` of baseline, objective/oracle agreement within `1e-12`, and zero target drift. This is a synthetic decoder experiment; it does not reproduce image masking, a ViT encoder, or MAE's benchmark results.

## 6. Running

Complete the two functions and select **Run tests**. The full suite and experiment run in CPU/WASM without external data or a pretrained network.

```bash
paperleet test representation/mae --impl problems/representation/mae/solution.py
paperleet mutate representation/mae
paperleet ci --problem representation/mae
paperleet vizdata representation/mae
```

## 7. Further reading

Read MAE Section 3 and the official `forward_loss` alongside its `patchify` and random masking pipeline. Why does MAE reconstruct only removed patches? How does target normalization change sensitivity to brightness and contrast? Why is sample variance different from the population variance used by Barlow Twins' BatchNorm?

Compare [BEiT](https://arxiv.org/abs/2106.08254), whose targets are discrete tokenizer IDs, and [iBOT](https://arxiv.org/abs/2111.07832), whose targets are soft online-teacher distributions. Each exercise isolates a different prediction target and reduction contract.

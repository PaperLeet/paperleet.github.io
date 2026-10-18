# I-JEPA Masked Latent Targets and EMA

[I-JEPA](https://arxiv.org/abs/2301.08243) predicts masked target representations from a context representation. Implement the target preparation, prediction loss and momentum update around that prediction. Target features and masks are supplied; the encoder and predictor are outside this exercise.

## Implement

### Return requirements

#### `masked_targets` (required)

Return a detached floating `Tensor` `[K*C*B,M,D]`, ordered by target block, context, then batch item. K is the number of masks, C is `num_context`, B is batch size, M is tokens per mask and D is feature width. Preserve the feature dtype/device; no gradient reaches teacher features and no input is modified.

#### `masked_prediction_loss` (required)

Return one scalar `Tensor`, shape `()`, of the common prediction/target dtype and device: smooth-L1 with beta=1 averaged over every tensor element. Retain prediction gradients but detach the target internally. Do not return per-token losses, a Python float or mutate either input.

#### `ema_update` (required)

Return a new detached `Tensor` with the same shape, floating dtype and device as `target` and `online`, containing the updated target parameters. Neither input receives gradients or is updated in place. This functional helper returns the new value, not `None`.

#### `_features` (provided)

Provided validator: return Python `None` when x is a finite, nonempty float32/float64 Tensor with the requested rank when ndim is supplied. Raise `ValueError` when a checked condition fails; do not mutate inputs or return transformed tensors.

#### `_pair` (provided)

Provided validator: return Python `None` when both finite nonempty floating tensors have identical shape, dtype and device. Raise `ValueError` when a checked condition fails; do not mutate inputs or return transformed tensors.

#### `_check_targets` (provided)

Provided validator: return Python `None` when features `[B,N,D]`, equal-length int64 masks `[B,M]`, positive integer context count and positive finite epsilon satisfy the target contract. Raise `ValueError` when a checked condition fails; do not mutate inputs or return transformed tensors.

### Formula and conventions

`masked_targets(features, masks, num_context=1, eps=1e-5)` receives teacher features `[B,N,D]` and a list of `K` target masks, each integer `[B,M]`. It must:

1. Detach the teacher features and apply layer normalization independently to each token's **D features**, without affine parameters. Use population variance and `sqrt(variance + eps)`.
2. Gather each batch item's target indices in the given order.
3. Repeat each target block's whole B-item batch `C=num_context` times, then concatenate target blocks. Return `[K*C*B,M,D]` in **target-block, context, batch** order.

For two target blocks, two contexts and two images, the leading-axis order is `(block0,context0,image0)`, `(block0,context0,image1)`, `(block0,context1,image0)`, `(block0,context1,image1)`, then block1 in the same order. Repeating each image consecutively would misalign the predictor's targets.

`masked_prediction_loss(prediction, target)` returns the mean smooth-L1 objective with `beta=1`:

$$
\ell(r)=\begin{cases}\tfrac12r^2,&|r|<1\\ |r|-\tfrac12,&|r|\geq1,\end{cases}\qquad
L=\operatorname{mean}\ell(prediction-\operatorname{sg}(target)).
$$

This follows the **released author's code** in `src/train.py`; the paper describes average L2 prediction distance. Both are identified explicitly here, and the judge requires smooth L1. The target is stopped even if the caller passes a tensor requiring gradients. Predictions remain differentiable. Residuals `[-2,0.5]` give loss `(1.5+0.125)/2 = 0.8125`.

`ema_update(target, online, momentum)` returns

$$\theta_{target}^{new}=m\operatorname{sg}(\theta_{target})+(1-m)\operatorname{sg}(\theta_{online}).$$

Return a new detached tensor and leave both inputs unchanged. This is a functional equivalent of the author's in-place parameter update. In training, the caller invokes it after the optimizer step, for each encoder parameter; the predictor is not included in the target-encoder EMA.

## Contract

Finite CPU float32/float64 tensors, equal dtype/device for paired tensors; preserve input values. Features are nonempty `[B,N,D]`; masks are int64 on the same device, have equal nonzero M, and contain valid token indices. Context count is a positive integer and epsilon positive and finite. Loss and EMA tensor pairs have identical nonempty shapes. Momentum is finite in `[0,1]`. Noncontiguous inputs are supported. Validators are supplied.

## Toy experiment

**Teaching adaptation.** Two synthetic six-token images supply latent features. Two target masks and two contexts exercise ordering. A small prediction table learns the selected targets through your loss. A separate fixed-online EMA example displays how target parameters approach online parameters. Scalar normalization/gather and the EMA closed form independently check the outputs. This does not train an image encoder or reproduce ImageNet pretraining. The preview uses the reference; T3 invokes all three submitted functions.

[Author training code](https://github.com/facebookresearch/ijepa/blob/52c1ae95d05f743e000e8f10a1f3a79b10cff048/src/train.py).

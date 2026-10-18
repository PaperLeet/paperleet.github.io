# BatchNorm: Batch Moments, Moving Estimates and Inference

## 1. What problem does the paper solve?

Batch Normalization introduces differentiable minibatch normalization and learned scale/shift. Training and inference use different sources of statistics. Implement [Algorithms 1–2 and Section 3.2](https://arxiv.org/html/1502.03167v3#S3) in a pure function. The title's internal-covariate-shift motivation is historical; this exercise does not establish a causal explanation of BN's effectiveness.

## 2. Core equations and return contract

For each channel, reduce the batch and all spatial positions, with `m = N * product(spatial)`:

$$\mu_B=\frac1m\sum_i x_i,\quad v_B=\frac1m\sum_i(x_i-\mu_B)^2,\quad y_i=\gamma\frac{x_i-\mu_B}{\sqrt{v_B+\epsilon}}+\beta.$$

The next estimates use the explicitly chosen EMA convention:

$$\mu'=(1-a)\mu_{run}+a\mu_B,\quad v'=(1-a)v_{run}+a\frac{m}{m-1}v_B.$$

Implement `batch_norm(...) -> (y, next_mean, next_var)`. Input is `[N,C,*spatial]`, rank at least two; affine parameters and buffers are `[C]`. Output matches the input; estimates are `[C]`. All tensors have matching float32/float64 dtype/device. Do not mutate any input. Returned estimates and inference statistics are detached; training normalization remains differentiable.

## 3. From equations to code

In training, compute moments over `(0,2,3,...)`. Use population variance for the output, but unbiased variance for the running estimate. Reshape channel quantities for broadcasting. In inference, use supplied running moments and return unchanged estimate values; do not recompute batch moments.

## 4. Exercise conventions

`momentum=a` is the **new batch's coefficient**, not the old state's coefficient. It is finite and lies in `[0,1]`; epsilon is finite, positive and inside the square root. Training requires at least two values per channel. Inference permits a single sample or empty batch. Channel/spatial widths must be positive. Invalid shapes, counts or parameters raise `ValueError`.

Unlike stateful framework layers, this exercise returns new buffers explicitly. Moving statistics must not retain computation graphs across optimizer steps. Inference must not differentiate into buffers, even if the caller supplied tensors with `requires_grad=True`.

## 5. Experiment

Eighty minibatches of 32 two-channel samples are drawn from a normal distribution with mean 3 and variance 4 (standard deviation 2), using seed 91. Adam with learning rate 0.08 learns scale 1.5 and shift −0.3 against an independent scalar normalization oracle; the graph also shows channel-zero moving estimates. Every output uses your implementation.

The checker requires final MSE below 0.003 and affine errors below 0.04. A fixed inference sample must have identical output when its companions change. This does not reproduce ImageNet, establish an acceleration factor or rank normalization methods by model accuracy.

## 6. Run your implementation

Complete `batch_norm` and select **Run tests**, or run:

```bash
paperleet test architecture/batch_norm --impl problems/architecture/batch_norm/solution.py
```

Checks include scalar values, spatial reductions, unbiased running estimates, analytical Jacobians, finite differences, detached state, strided tensors, empty inference and actual affine training.

## 7. Further exploration

Compare BN's training dependence on other examples with the sample-local reduction of GroupNorm. Derive the fixed inference affine transform. Investigate whether small batches change the variance of estimates before drawing any claims about downstream model quality.

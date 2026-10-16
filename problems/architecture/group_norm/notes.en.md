# Reasoning about GroupNorm

The reduction set is the algorithm. A group contains consecutive channels of one sample and every spatial position of those channels. Reshaping only exposes this set; it must not mix samples or use a different interleaving of channels.

For a scalar loss, let $u_i$ be its upstream derivative, $z_i=(x_i-\mu)/s$, $s=\sqrt{v+\varepsilon}$ and $q_i=u_i\gamma_{c(i)}$. Inside one group of size $m$:

$$
\frac{\partial L}{\partial x_i}
=\frac{q_i-\operatorname{mean}(q)-z_i\operatorname{mean}(qz)}s.
$$

The means here use that same sample/group. Channel-scale gradients sum $u_i z_i$ over batch and spatial positions; channel-bias gradients sum $u_i$ over those positions. A detached mean or variance can produce correct forward values while failing this Jacobian.

For one value per group, centering makes the value zero and the two direct/mean gradient paths cancel. Epsilon makes the denominator finite; it does not turn the variance denominator into $m-1$.

`G=1` and `G=C` change the moments, while affine parameters remain per channel in both cases. The former therefore shares layer-wide statistics with LayerNorm, but a generic LayerNorm layer may have differently shaped affine parameters.

The companion-sample experiment tests a structural property of GN. Its batch-statistics comparison is computed from current activations, without running averages. It illustrates batch dependence during training, not an accuracy ranking of trained networks. The [paper's formulation](https://arxiv.org/html/1803.08494v3#S3) and [official operator contract](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GroupNorm.html) are the primary references.

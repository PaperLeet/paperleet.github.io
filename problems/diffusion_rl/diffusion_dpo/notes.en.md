# Solution notes

Reshape cumulative alpha to `(B, 1, ..., 1)` and use its square root for the signal and the square root of its complement for the noise. Sharing noise gives the useful invariant

$$
x_t^w-x_t^l=\sqrt{\bar\alpha}(x^w-x^l).
$$

Flatten all non-batch coordinates before taking the error mean. Subtract the rejected error from the preferred error separately for current and reference models. A smaller current gap than reference gap must produce a positive margin, explaining the negative sign before `beta/2`. `softplus(-margin)` is stable even when a confidently wrong pair has an enormous positive loss.

For pair $i$ with $D$ coordinates and $s_i=\sigma(-m_i)$, the gradients are

$$
\frac{\partial L}{\partial\epsilon^w_{\theta,i,d}}
=\frac{\beta s_i}{BD}(\epsilon^w_{\theta,i,d}-\epsilon_{i,d}),
\qquad
\frac{\partial L}{\partial\epsilon^l_{\theta,i,d}}
=-\frac{\beta s_i}{BD}(\epsilon^l_{\theta,i,d}-\epsilon_{i,d}).
$$

Reference predictions and target noise define the comparison and must not move in response to this gradient. Detach them explicitly. Feature averaging makes the loss invariant to repeated coordinates, and averaging pairs gives the same invariance when a batch is duplicated.

The toy affine denoiser can reduce this loss by increasing the preferred-versus-rejected margin; absolute winner reconstruction error is not guaranteed to decrease at every update. This distinction follows directly from optimizing an error difference.

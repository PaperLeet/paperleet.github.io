# Solution notes

Generation moves from high noise to low noise, so `dt = next_time - time` is negative. The mean uses this signed step; the noise standard deviation uses its positive magnitude. Both the state and velocity terms in the score correction are necessary. Substituting the chosen noise schedule simplifies the velocity derivative:

$$
\frac{\partial\mu}{\partial v}=\Delta t\left(1+\frac{a^2}{2}\right)I.
$$

For the loss, center rewards along the group axis only, then compute the population variance of those centered values. Add epsilon after the square root. Detach this advantage before applying it to every time step.

All three Gaussian policies share a variance. Subtract their squared residuals to compute the log ratio directly; the Gaussian normalizers cancel. Sum over features before exponentiating. Do not average time steps before forming ratios, since the clipped objective operates on each transition.

The minimum in the surrogate handles advantage signs. A positive advantage stops gaining surrogate reward above the upper ratio boundary; a negative advantage stops gaining below the lower boundary. Replacing the minimum with a maximum defeats that constraint. The exact KL contributes a nonnegative mean-distance penalty even when all rewards are tied.

At the old policy, before clipping and with $N=PGT$, the gradient is

$$
\frac{\partial L}{\partial\mu_{pgtd}}=
\frac{-A_{pg}(a_{pgtd}-\mu_{pgtd})+\beta(\mu_{pgtd}-\mu_{{\rm ref},pgtd})}{Nq_t^2}.
$$

Samples, rollout means, reference means and variances must remain fixed for this score-function derivative. Backpropagating through the sampled action changes the estimator and can cancel the policy gradient.

For numerical stability, clip the log ratio before exponentiating: cap at `log(1 + clip_epsilon)` for positive advantages, floor at `log(1 - clip_epsilon)` for negative advantages, and use zero log ratio for zero advantages. This is algebraically equivalent to the clipped surrogate and avoids `infinity * 0`.

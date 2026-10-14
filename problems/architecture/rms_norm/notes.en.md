# Reference notes

RMS is the square root of the mean square, not variance or an unscaled L2 norm. Do not center, apply a sample-variance correction, or include affine weights in the statistic. Retain the reduced dimension for token-local broadcasting.

Epsilon placement is a numerical contract. This exercise uses the inside-square-root version; the authors' older outside-epsilon code is an alternate convention, not a paper mistake. A zero vector has zero output but input Jacobian `diag(weight)/sqrt(eps)`, so returning a detached constant loses a real gradient.

With zero-based indices and $s=\sqrt{\sum_{j<k}x_j^2/k+\varepsilon}$:

$$
\frac{\partial y_i}{\partial x_j}
=w_i\left(\frac{\delta_{ij}}s-\frac{x_i x_j}{k s^3}\mathbf1_{j<k}\right).
$$

Suffix inputs do not affect the statistic, but losses on suffix outputs still affect prefix inputs through the shared denominator. Detaching that statistic keeps forward values while dropping the second term; analytical VJPs and finite differences expose the error.

Only `k=D` must equal full normalization. An unrepresentative, low-energy prefix can strongly amplify the suffix. The synthetic plot deliberately constructs that case and makes no general performance claim about partial normalization.

For positive epsilon, the exact identity is `f(a*x, a*a*eps)=f(x, eps)` for $a>0$, rather than scaling with fixed epsilon. Input shifts are not an invariance either.

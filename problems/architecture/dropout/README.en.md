# Inverted Dropout: Explicit Masks and Expected Activations

## 1. What problem does the paper solve?

Dropout randomly changes which units participate in training. Section 10 also describes the inverted convention: scale retained activations during training and keep inference unchanged. This exercise makes the keep mask explicit so scaling, endpoints and gradients can be checked directly.

[原始来源 / Original source](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) · Section 4; Section 10, page 1951: inverted Bernoulli scaling and its mean/variance.

## 2. Core equations and return contract

$$q=1-p,\quad M_i\sim\operatorname{Bernoulli}(q),\quad Y_i=X_iM_i/q,\quad\mathbb{E}[Y_i]=X_i,\quad\operatorname{Var}(Y_i)=X_i^2p/q.$$

Required functions: `inverted_dropout`.

## 3. From equations to code

At stochastic training sites multiply kept values by `1/(1-p)` and output zero elsewhere. Handle inference and p=0/p=1 explicitly, avoiding division by zero while retaining a connected zero gradient.

## 4. Exercise conventions

x is a finite float32/float64 tensor of any shape, including scalars, empty dimensions and strided views. **p is the drop probability**, a finite real number in `[0,1]`; the paper also uses p for keep probability, so translate that notation.

For training with `0<p<1`, mask must be a bool **keep** mask with exactly the same shape/device as x. No broadcasting or internal sampling: True retains and rescales, False drops. At p=0 or in inference, ignore mask and return equal input values. Inference remains identity even at p=1. Training at p=1 returns graph-connected zero with zero gradients and needs no mask. Preserve dtype/device/shape and input gradients without mutating x or mask; fresh storage is not promised. Invalid probabilities/dtypes/stochastic-training masks raise ValueError.

## 5. Experiment

The harness supplies 64×4 Gaussian features, a fixed linear target and trainable linear weights. Your dropout transforms training inputs at p=0.25 for 100 Adam updates (lr=0.06, seed=193, float64); the chart evaluates MSE without dropout. Another 8192 realized masks measure mean/variance errors against the analytic moments. Acceptance requires MSE<20% of initial, maximum mean error<0.06 and variance error<0.08. This demonstrates behavior, without claiming dropout always improves generalization.

## 6. Run your implementation

Complete the starter and run all tests in the workspace to unlock Notes. Maintainers can also run:

```bash
paperleet test architecture/dropout --impl problems/architecture/dropout/solution.py
```

## 7. Further exploration

Derive the fixed-mask input Jacobian, then marginalize the mask to obtain both moments. Compare the paper's test-time scaling convention without mixing the two parameterizations.

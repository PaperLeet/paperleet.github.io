# SigLIP solution

Normalize the two inputs along their feature axis. `image_unit @ text_unit.T` gives all pairwise cosine similarities. Multiply by `log_scale.exp()` and add `bias`; both calibration inputs must remain tensors in the computation graph.

Construct signed labels with `2 * eye(B) - 1`. Each pair contributes `softplus(-label * logit)`. Sum the entire matrix and divide by $B$. A stable softplus implementation handles large logits without the overflow of a direct `log(1 + exp(x))` expression.

Let $y_{ij}$ be one on the diagonal and zero elsewhere. Then

$$
\frac{\partial L}{\partial\ell_{ij}}=\frac{\sigma(\ell_{ij})-y_{ij}}{B}.
$$

The negative gradient on a positive pair raises its logit during gradient descent. The positive gradient on a negative pair lowers it. This formula verifies signs, reduction and saturated inputs independently of the reference implementation.

The bias gradient sums all logit derivatives. The log-scale gradient additionally multiplies each derivative by the cosine similarity and `exp(log_scale)`. Both feature towers receive gradients through their row normalization and dot products. Calling `.item()` or `.detach()` on these values would disconnect part of the model.

Positive rescaling of individual feature rows leaves logits unchanged. Applying the same permutation to both paired batches merely permutes the loss matrix, and transposing that matrix preserves its sum. These invariants catch accidental batch coupling.

The experiment optimizes four feature pairs and two calibration parameters. It illustrates the objective with actual gradient updates and has no claim about full encoder pretraining or paper benchmark accuracy.

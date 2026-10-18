# Implementation notes

This is the penalty in EWC Eq. (3), with the Fisher diagonal supplied as input. Do not substitute newly squared parameter gradients or a batch mean for the supplied Fisher, and do not add the new-task loss inside this function.

\[
\nabla_\theta P=\lambda F\odot(\theta-\theta_A^*).
\]

Both `anchor` and `fisher_diag` are fixed records of the old task. Training the anchor moves the constraint toward the current model. Detaching the current parameters instead produces the correct value but no useful regularization gradient.

Typical mistakes are dropping `1/2`, averaging parameters, ignoring unequal Fisher weights and allowing gradients into the anchor. Penalties for independent parameter blocks must add; this distinguishes a sum from a mean.

For zero strength, form elementwise zeros connected to `theta` before reducing, avoiding overflow in an unnecessary difference or square. Mask zero-Fisher displacements before squaring: `0 * inf` is not zero. Empty-input sums must remain differentiable.

## Independent experiment check

Task A learns an anchor near `(2,0)`. B's data objective is `1/4 * ((theta0+2)^2 + (theta1-2)^2)`, with Hessian `0.5*I`. Given `F=(2.5,0)` and `lambda=1`, the exact optimum is `((0.5*(-2)+2.5*anchor0)/3, 2)`, approximately `(4/3,2)`. Tests compare actual SGD updates against that independent solution, without using the reference function to generate labels or the optimum.

Without regularization the solution is near `(-2,2)`, giving old-task MSE near 40. EWC gives old-task MSE near `10/9` and new-task MSE near `50/9`. Some forgetting remains and the new task is not fitted perfectly. This measures a tradeoff, rather than establishing retention for every task sequence.

Source: [original paper, Section 2, Eq. (3)](https://arxiv.org/html/1612.00796v2#S2.E3). The two-parameter Gaussian linear model, supplied Fisher and deterministic data are this exercise's teaching adaptation.

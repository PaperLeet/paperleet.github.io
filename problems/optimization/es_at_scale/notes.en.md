# Implementation notes

Center rewards, compute their population RMS deviation, clamp the resulting standard deviation, and broadcast one weight per candidate. Sum over the population and divide by its size. The ascent sign matters because rewards are maximized.

`noise` is epsilon, not sigma times epsilon. The update convention absorbs inverse sigma into alpha. Adding another sigma factor changes the optimizer. Reward standardization is not a softmax or a rank transformation.

Detaching all observations and weights is an explicit optimizer contract. It prevents an accidental graph through rollouts or prior updates. Constant rewards, including a singleton, must return finite zeros. A positive affine reward transformation preserves well-resolved standardized rewards; this invariance need not hold when the standard-deviation floor is active.

The teaching search uses a known optimum and evaluates actual candidate rewards. The 0.08 acceptance bound leaves margin around the measured seed-42 distance of 0.01254; it is not a general convergence theorem. No pretrained-model scaling claim follows from this two-coordinate task.

Source: [Section 3, Algorithms 1–2](https://arxiv.org/abs/2509.24372). Layer-wise seed regeneration and distributed inference are outside this exercise.

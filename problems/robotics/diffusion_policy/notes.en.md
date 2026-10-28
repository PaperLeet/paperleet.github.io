# Implementation notes

The square roots convert variances into standard deviations. A `(B,)` schedule must reshape to `(B,1,1)`; ordinary right-aligned broadcasting can accidentally attach it to the action axis. The epsilon loss is an elementwise mean, so duplicating action coordinates does not change it. Stop the target noise, not the prediction.

Three horizons describe different windows: observation history `To`, predicted actions `T`, and executed actions `Ta`. Under the stated observation/action alignment, the current action is `To-1`; starting at `To` loses the first actionable prediction. The past and unexecuted suffix must have zero slice gradient.

The experiment's goal-conditioned basis makes the noise predictor analytically identifiable at a fixed noise level. Its convergence checks the training mechanism, while independent examples and gradients check the mathematical contract. Accurate action reconstruction here is not a robot success rate.

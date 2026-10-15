# Implementation notes

Sum squared entries across every parameter, then compute one norm and one scale. Return the original norm, not the clipped norm. Never amplify a gradient that is below the limit.

Common mistakes:

- Clip each parameter separately.
- Use elementwise value clipping.
- Amplify gradients below the threshold.
- Report the norm after clipping.

## A global scale preserves direction

For parameter gradients `[3]` and `[4]`, the global norm is 5. A maximum of 2.5 multiplies both by `2.5/(5+eps)`. Per-parameter clipping uses different scales and changes the joint gradient direction.

Elementwise clamp is also different from norm clipping. Below the limit, scale must stay 1. Return the pre-clipping norm for diagnostics, not the norm after scaling.

In a training loop, complete accumulation first; if loss scaling is used, unscale before one global clipping step, then update parameters. This exercise receives ready-to-update gradients and has no mixed-precision state. For P total parameter elements, time and output space are O(P).

For extreme finite gradients, normalize by the largest absolute element before squaring in float64; restore the scale afterward. This avoids overflow/underflow of naive squares when the norm itself is representable. Scale the denominator too, so epsilon remains effective for tiny gradients.

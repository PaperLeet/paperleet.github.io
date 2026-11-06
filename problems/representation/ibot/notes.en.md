# iBOT implementation notes

This module follows the same-view patch component of official `iBOTLoss.forward`. The complete objective additionally uses image-level cross-view CLS self-distillation and averages appropriate crop terms. The full training system updates teacher weights and centers by moving averages; those state transitions are intentionally not hidden inside this loss. The fixed `[K]` center is a simplified broadcast shape for the official patch center.

Teacher centering subtracts prototype-specific biases before temperature-softmax. Detach the resulting distribution, not merely the logits before subtracting a live center. The gradient boundary covers both teacher and center. Student temperature contributes a `1/student_temp` chain-rule factor. Soft teacher targets retain uncertainty: when the student matches them, CE equals entropy and KL is zero.

| Fault | Independent evidence |
| --- | --- |
| Missing center | Nonconstant prototype offsets change the independently computed target probabilities. |
| Missing teacher temperature | Tests exercise sharp default and softer explicit temperatures. |
| Teacher or center gradient leak | Target tensors must be stopped, loss gradients absent, and both tables fixed during optimization. |
| Missing student temperature | Exact student gradients include its chain-rule factor. |
| Global masked mean | Images with `1,2,0` selected patches distinguish the official image-balanced reduction. |
| Including visible patches | Their score changes cannot affect loss; their student parameters must never move. |
| Wrong image correspondence | Asymmetric teacher examples expose cross-image target pairing. |
| Hard teacher labels | Scalar soft targets, entropy-floor values and zero gradient at distribution matching rule out argmax. |
| Strided student gradient loss | Interleaved parent VJPs preserve all prototype derivatives while teacher/center parents remain stopped and unused storage stays zero. |

Empty mask rows contribute zero and remain in the batch denominator, exactly as the released clamped per-image formula specifies. The single-prototype and large-score tests separately exercise endpoints and numerical stability. Central differences use independent scalar softmax/CE arithmetic; only the student is perturbed because teacher targets are defined as stopped.

T3 uses 48 student logits with mask counts `1,3,0`, 120 Adam updates at `.08`, and temperatures `.7/.5`. PyTorch 2.14.1 CPU float64 measured KL `.10580984 → 1.80e-7`, CE `.81916371` and teacher entropy `.81916353`. Acceptance requires KL below `1e-5` and `.001` of its baseline, plus zero teacher/center/visible drift. The common fixture uses seed 42 and two threads; the experiment has deterministic sinusoidal initial conditions. Fixed-teacher logit fitting demonstrates only this objective, not emergence of useful representations or online-tokenizer training.

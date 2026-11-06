# MAE implementation notes

The core is a reconstruction criterion on already patchified pixels. MAE's asymmetric encoder, lightweight decoder, random removal/restoration indices and pretrained representation quality lie outside this module. The official `models_mae.py` normalizes each patch's target using `torch.var` with its default sample correction, so the divisor is `P-1`. Using population variance is a coherent alternative normalization, but it violates this exercise's released-code convention. The term “pixel” includes every flattened channel coordinate.

The reference normalizer preserves its own gradient; the loss calls it on detached data. Official image tensors normally do not require gradients, so this explicit interface boundary does not change their usual training behavior. Empty masks are an exercise extension: a clamped denominator returns differentiable zero without pretending an unobserved patch contributes evidence.

| Fault | Why a check catches it |
| --- | --- |
| Population variance | The two-coordinate `[-1,1]` example separates divisors 1 and 2. |
| Patch-axis normalization | Independent scalar values and per-patch translations identify the wrong axis. |
| Inverted mask | Visible-only changes must have no effect; selected prediction gradients are analytic. |
| All-patch averaging | Unequal masks give a known global mean over exactly three patch errors. |
| Pixel sum | A two-pixel patch must average its coordinate errors. |
| Target gradient leak | The loss's target gradient must be absent; the training target cannot drift. |
| Ignored normalization option | Raw and normalized targets have separately computed scalar objectives. |
| Stopped prediction | The scalar must remain connected to predicted pixels. |
| Strided input gradient loss | Interleaved parent VJPs validate raw/normalized reconstruction and the standalone normalizer, including exact zeros in unused storage. |

The standalone normalization VJP is checked against finite differences of Python scalar arithmetic, not the reference function. T3 trains 16 shared decoder parameters on 24 deterministic contexts, supervising 14 patches with 120 Adam steps at `.06`. In native PyTorch 2.14.1, CPU float64, the baseline MSE `.67053234` fell to approximately `6.57e-6`, below the declared `1e-4` and relative `.001` thresholds. The common fixture fixes seed 42 and limits PyTorch to two threads; the experiment itself uses no random draws. Target inclusion in the optimizer detects an incorrect movable-label shortcut. This is training-set reconstruction, not held-out image quality.

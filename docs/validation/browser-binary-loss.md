# Binary logits loss browser support

The browser backend adds the functional
[`binary_cross_entropy_with_logits`](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.binary_cross_entropy_with_logits.html)
and [`BCEWithLogitsLoss`](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BCEWithLogitsLoss.html)
module API. The module registers constant weight/positive-weight buffers and has
no trainable parameters. This fills an actual compatibility gap: a SASRec learner
using these APIs already passes the complete ten native tests, while the
committed previous browser adapter raises `BrowserUnsupported`.

## Scope and numerical implementation

Matching float32/float64 logits and target shapes are supported, including
scalars, strided arrays and empty axes. Constant weights of the same dtype may
broadcast to the input shape. Both positive and negative terms use stable
log-sigmoid/log-partition values. Mean reduction divides by the number of output
elements, including zero-weight elements, rather than the sum of weights.
No reduction, sum and mean are supported; empty mean is NaN and empty sum is zero.

The raw NumPy oracle has analytical VJPs for both logits and targets, with saved
storage guards including frozen weights. Native CPU Inf/NaN behavior is retained
instead of clamping values or suppressing invalid results. Finite JVP supports
both input channels, and reverse differentiation through the primal/tangent
retains actual graphs. Legacy arguments, mixed/differentiated weights, nonfinite
JVP and generic higher-order backward remain on the server. Ordinary malformed
shapes/reductions are learner errors. The adapter remains a bounded teaching
subset, not native PyTorch or a claim to support every loss API.

## Independent checks

Forty-one focused tests pass in 3.50 seconds. They compare float32/float64 values,
input/target gradients, reductions, scalar/strided/empty shapes and broadcast
sample/positive weights against native PyTorch 2.14.1. Separate scalar Bernoulli
likelihood and central differences check both gradients independently. Further
checks cover extreme logits, native nonfinite targets/weights, zero weights,
out-of-range labels, finite JVP/reverse-over-forward, empty JVP, no-grad calls,
weight mutation guards, module buffers and installed public API dispatch.

A learner using the module API for its positive term and functional API for the
negative term passes all ten native SASRec checks and the same ten unchanged
exported NumPy checks, including all 160 training updates. Replacing negative
labels with positive ones is independently rejected. No native or browser suite,
precision tolerance, experiment budget or calibration mutant is reduced.

Only the shared NumPy autodiff adapter changes. Exported manifests/details for
its 66 dependent modules receive new immutable hashes; Schedules is unchanged.
Every other index field, all paper/category records and original lesson source
are unchanged. Older committed immutable assets remain available to cached
manifests.

## Full regression and real browser

- Public-only regression: **1,160 passed, one optional Docker skip**, 129.67
  seconds. The existing TorchScript deprecation warning remains. This includes
  complete per-module reference/fault calibration in isolated NumPy adapter
  subprocesses; it is not a claim of running every module in a real browser.
- Complete frontend checks and standalone production build pass in an isolated
  snapshot excluding unrelated in-progress UI edits. Initial JavaScript remains
  **399,721 bytes raw / 128,524 gzip**, within the original 400 KiB / 130 KiB caps.
  This Python operator lives in the on-demand shared runtime, not the main JS.
- Real Pyodide CPU/WASM: the alternate SASRec learner passes **10/10**, with all
  seven maintained faults calibrated and all 160 actual training updates. A
  module-level QA probe additionally checks weighted JVP primal/tangent and
  reverse gradients of both primals/tangents against independently computed
  native float64 constants. Its assertions execute before the unchanged suite.
- Initial observed phases: preparation **1.62 s**, calibration **1.56 s**, complete
  tests **0.23 s**. These single localhost observations are not internet cold-start
  or Oracle hardware comparisons. The learner Lab displays its own 160-step
  curve and history-dependent training metrics.
- Reversing negative labels fails at **4/10**. Restoring correct labels returns
  **10/10**. A legacy `reduce=False` call displays the unsupported-API message,
  preserves source and marks the old successful report as older code; this
  standalone QA site has no server configured. The generic hosted fallback is
  unchanged, but no new live hosted fallback submission is claimed here.
- Final browser warning/error logs are empty and the static QA server records
  zero API requests or POSTs. Temporary tabs/server/snapshot are cleaned up;
  raw evidence remains under ignored `reports/bce-*` paths.

No public push, production app/image deployment or account credit is claimed.
Private pinned-core integration and native hosted checks are recorded separately.

# Browser L1 pairwise distances

Learner implementations may now use `torch.cdist(x1, x2, p=1)` in the bounded
NumPy/WASM adapter instead of requiring the native server for that operation.
This expands supported implementations within the existing **66/66 modules**;
it does not add a module or imply complete PyTorch compatibility. The catalog
still has **517 maintained checks and 272 maintained mutants**.

## Semantics and bounds

L1 distances sum absolute feature differences. Reverse-mode derivatives use
the sign of each difference, with zero at tied coordinates; each operand's
gradient is reduced over pairwise and broadcast batch axes. NaN signs propagate
as in native CPU PyTorch, including `inf - inf`. Values, dtype and analytical
gradients are retained for float32/float64, empty point/feature/batch dimensions
and strided inputs. All three accepted compute modes remain valid for L1;
they only select Euclidean strategies in native PyTorch.

The existing 16,000,000-element pairwise allocation bound is unchanged and also
bounds zero-width feature output sizes. Other orders, including `p=0`, `p=.5`,
`p=3` and `p=inf`, remain unsupported in this adapter. Forward-mode cdist is
still explicitly unsupported. Gradient disabling and mutation checks remain
in force. There is no finite-difference gradient implementation.

The intended subset follows the official [cdist contract](https://docs.pytorch.org/docs/2.14/generated/torch.cdist.html)
and the native [CPU distance kernel](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/cpu/DistanceOpsKernel.cpp).
Those sources describe a broader native implementation; this adapter supports
only the bounded subset above.

## Evidence

- Public Python regression suite: **1118 passed, one skipped**, 124.12 seconds.
  This includes unchanged complete reference/mutant browser calibrations for
  every maintained module. The skip is the explicitly opt-in Docker submission
  harness. One existing upstream TorchScript deprecation warning remains.
- L1/L2 native parity checks cover both precisions, batch broadcasting, empty
  inputs, strides, compute modes and memory limits. Independent L1 checks use
  hand-calculated distances and asymmetric gradients for two distinct leaves,
  plus scalar Python values and central finite differences for both operands
  away from ties. Nonfinite and overflow gradients are compared with native
  PyTorch 2.14.1. Existing Euclidean tolerances are unchanged.
- Full frontend checks and isolated standalone production build passed. Initial
  JavaScript remains **397,494 raw / 127,713 gzip bytes**, within the original
  400 KiB / 130 KiB limits. The build excludes concurrent uncommitted UI work.
- Real browser, self-hosted Pyodide **0.29.4** / NumPy **2.2.5**: Attention
  **7/7** with all **three** faulty implementations caught; LoRA **6/6** with
  all **four** caught. Disposable learner fixtures add float32/float64 L1
  value, tied-coordinate gradient and nonfinite-gradient assertions before
  executing each unchanged full suite, including LoRA's original training.
- A disposable `p=3` source shows the unsupported-API notice in browser-only
  mode and keeps the earlier result marked as older source. Restoring the L1
  fixture reruns full calibration and returns Attention **7/7**. Hosted automatic
  fallback code is unchanged; no new hosted/native fallback run is claimed here.
- The isolated production server recorded **96 requests, zero API requests,
  zero POSTs and zero HTTP errors**. Browser warnings/errors were empty.
  Screenshots and disposable fixtures are retained locally under ignored
  `reports/l1-cdist-*` paths. Timings in the UI are localhost observations,
  not internet cold-start or Oracle performance measurements.

No problem statement, test, reference solution, mutant, experiment, training
step or threshold changed. Exported catalog/detail manifests now reference a
new content-addressed shared runtime. All 1,718 previously tracked data assets
remain available for older clients. Full hydrated-bundle hashing invalidates
compatibility-proof reuse when the adapter changes, even if a URL were reused;
the existing frontend regression verifies this boundary. Learner executions
still start fresh isolated workers and run complete tests.

Earlier records describing `p=1` as unsupported are historical; `p=3` now serves
as an unsupported-order fixture. No public push or production release is
included in this iteration. The preceding approved Nginx-only deployment did
not publish these runtime changes.

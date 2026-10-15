# Browser cosine similarity with native derivative semantics

The CPU/WASM adapter adds `torch.cosine_similarity`,
`torch.nn.functional.cosine_similarity` and `torch.nn.CosineSimilarity`.
The public contract is checked against the [PyTorch functional API](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.cosine_similarity.html)
and independent native PyTorch 2.14.1 execution. This implements the API, not a
new CLIP task or a change to any lesson's tests, references or training budget.

## Numerical and differentiation scope

Real float32/float64 vectors support positive/negative axes, scalar and empty
vectors, noncontiguous views, broadcasted batches, mixed floating precision and
integral/bool tensors paired with a floating tensor. Both integral inputs are
an ordinary native-compatible error. Unsupported half precision, complex
results and nested forward AD retain their existing bounded behavior; complex
cosine itself is invalid in native PyTorch. Epsilon may be zero or infinity;
negative and NaN values are ordinary errors. Nonfinite outputs and gradients
are preserved rather than converted into passing finite values.

The reverse derivative near epsilon is deliberately not implemented as a product
of two `F.normalize` calls. Native cosine clamps cloned norm values under
`no_grad`, retaining their identity reverse edges. Native forward AD changes the
norm tangent to zero below epsilon and half at a tie; the tangent's reverse edge
also stays identity. The adapter explicitly retains these separate value/edge
rules. As a result, the JVP below epsilon need not equal a dot product of the
native reverse gradient with the direction. Both channels agree with native,
including reverse gradients with respect to the original inputs and tangent
inputs. At a zero vector the native reverse-over-forward primal gradient can be
NaN; this behavior is retained, not replaced with a convenient zero.

## Independent verification

There are 46 new parameterized checks covering both inputs' VJPs, default and
large epsilon neighborhoods, zero/empty/scalar inputs, mixed precision and
integral promotion, broadcasting, noncontiguous storage, aliases, frozen inputs,
saved-tensor mutation, extreme/nonfinite values, invalid arguments, functional
aliases/module registration, JVP and reverse gradients through both channels.
The new checks plus existing autodiff, forward-mode, binary-loss and KL-loss
suites pass **316 tests in 5.32 s**. Native TorchScript emits its existing
FutureWarning. The installed API test runs in a subprocess so the adapter never
replaces native torch inside the independent comparison process.

The complete public `pytest -q tests` regression passes **1,260 tests, one
optional Docker skip**, in **163.05 s**. This includes unchanged complete module
reference/mutation calibration against the exported NumPy adapter; that native
NumPy calibration is separate from the actual WASM evidence below. Concurrent
account, UI and publication-policy work is excluded from the source snapshot.
No tolerance, reference, mutation fixture or test/training budget is relaxed.

## Actual browser and native fallback

A fresh static build runs the new adapter in actual Pyodide CPU/WASM:

- A separate CLIP learner using `F.cosine_similarity`, broadcasted image/text
  vectors and an independent symmetric `log_softmax` objective passes **7/7**.
- The module form `nn.CosineSimilarity` also passes **7/7**. All four maintained
  wrong implementations are calibrated and all **80 actual optimization steps**
  execute; these are the existing paired-vector teaching task, not encoders or
  a paper-scale benchmark.
- Replacing cosine normalization with raw dot products returns **4/7**. Restoring
  the module API passes **7/7** again.
- Appending a valid native-only matrix 1-norm call produces the unsupported-API
  notice, retains the previous result as stale source, and preserves the exact
  **820-character** source through the download control. The temporary download
  is removed. Restoring correct code passes again.
- First observed localhost phases are preparation **1.53 s**, calibration
  **0.42 s**, full learner tests **0.07 s**. These are not internet cold-start
  measurements or comparisons with Oracle hardware.

The CLI's native Torch judge separately accepts the functional alternate learner
at 7/7. Private packaged-worker validation is recorded with the pinned core;
no live authenticated server fallback or new Docker image is claimed.

## Resource identity and release boundary

The reviewed export creates **135 new immutable assets**: one shared autodiff
runtime plus 67 manifests and 67 details. The Schedules runtime is unchanged.
Catalog/index changes are only the new browser/detail references; all 68 modules,
53 paper records, lesson tests/references and 2,123 prior data files are retained.
Old immutable files remain available. No shared worker, sandbox protocol, Python
package download, account, community or server implementation is changed.

The full frontend checks and standalone/hosted builds pass. Their initial graphs
remain unchanged at **401,047 raw / 129,981 gzip bytes** standalone and
**406,423 raw / 131,823 gzip bytes** hosted, below the original 400 KiB / 130 KiB
caps. The new operator is carried in the shared Python runtime asset, not the
initial JavaScript shell.

The actual module learner's displayed training artifact has initial loss
1.9918, final loss 0.00020634 and both retrieval accuracies 1.0. The browser
warning/error log is empty; the static server records 114 requests, zero API
requests/POSTs and no HTTP errors. Temporary tabs, verification download and
server are closed; the isolated build snapshot is removed after committing.
Ignored evidence and a screenshot of 7/7 plus the actual loss curve are retained.
No push or production application/image deployment occurs; the previously
approved Nginx proxy release is unchanged.

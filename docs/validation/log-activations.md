# Softplus and LogSigmoid browser modules and forward AD

BPR and SASRec learners can use `nn.Softplus` and `nn.LogSigmoid`, while their
functional forms support JVP and reverse gradients through both JVP channels.
This changes the CPU/WASM adapter, not any problem contract, reference, lesson
test, mutant or training budget. The public signatures and threshold contract
come from the official [Softplus API](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.softplus.html)
and [LogSigmoid API](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.logsigmoid.html);
all numerical and derivative behavior is checked against native PyTorch 2.14.1.

## Concrete compatibility failures

Previously module forms were unavailable and JVP raised the explicit unsupported
operation notice. Two native-semantic discrepancies also existed in Softplus:

- A float64 input of 1000 with threshold 1000 returned finite 1000 in the adapter,
  whereas native `log1p(exp(x))` overflows because the strict linear threshold is
  not crossed. The native first derivative is NaN in that case.
- Native CPU backward multiplies the incoming gradient by `exp(beta*x)` before
  dividing by `exp(beta*x)+1`. Precomputing a bounded slope and then multiplying
  can incorrectly hide native overflow for a float32 incoming gradient of 1e38
  at input 10. Both VJP and JVP must preserve that operation order.

The new implementation uses the native threshold/value branch and incoming
VJP/JVP order. The second derivative uses the native sigmoid saturation and
strictly-below-threshold mask: at a threshold tie the first derivative remains
nonlinear while the curvature is zero. Negative and signed-zero beta, infinite
or NaN beta/threshold values, scalar/empty/noncontiguous inputs and native
nonfinite results are retained. Invalid non-floating inputs are ordinary errors;
unsupported half precision and tensor-valued Softplus parameters retain the
server path. Nested forward AD and general higher-order reverse AD remain bounded.

LogSigmoid's reverse-over-forward curvature uses `sigmoid(x)*(sigmoid(x)-1)`.
Differentiating the complementary sigmoid instead would saturate on the wrong
tail. Explicit zero-absolute-tolerance tail checks prevent broad error bounds
from concealing that difference. The two new modules delegate to these shared
implementations and support the existing sequential module container.

## Independent numerical evidence

The new file adds **63 parameterized checks** of values, first VJPs, JVPs,
reverse gradients of original inputs and tangent inputs, native threshold ties,
zero/negative/nonfinite parameters, extreme incoming gradients, scalar/empty
shapes, precision, storage layout/version guards, ordinary errors, explicit
fallback and installed functional/module APIs. These plus the existing autodiff,
forward, tensor, binary-loss, KL-loss and cosine suites pass **408 checks in
6.61 s**. The native TorchScript deprecation warning remains.

The final public `pytest -q tests` regression passes **1,323 tests, one optional
Docker skip**, in **161.93 s**. It includes unchanged module reference/mutation
calibration against the exported NumPy adapter. That calibration uses a native
NumPy process and is distinct from the actual WASM evidence below. An initial
complete run preceded the extreme-incoming-gradient correction; only the final
run is used as evidence for the committed source.

## Actual browser and source recovery

A static production build is exercised in real Pyodide CPU/WASM:

- SASRec using `nn.Softplus` passes **10/10**, calibrates all seven maintained
  wrong implementations and executes all **160 actual updates**. Its displayed
  training accuracy reaches 1.0 and loss falls from 1.3863 to 0.023313.
- A wrong paired objective returns **6/10**, while restoring the binary objective
  returns **10/10**. Existing tests, scalar oracles and tolerances are unchanged.
- BPR using `nn.LogSigmoid` passes **11/11**, calibrates all eight maintained
  wrong implementations and executes **120 actual updates**.
- Both learner sources include six independently native-generated operator
  probes, spanning float32/float64, both activations, threshold ties and huge
  incoming gradients. The **24 expected tensor checks** verify primal values,
  JVPs and reverse gradients of both input/tangent channels in actual WASM.
  Expected arrays come from native Torch, not another adapter expression.
- The complete 9,239-character SASRec source, including those probes, is
  recovered exactly through the download control. A BPR source with a valid
  native-only matrix 1-norm call produces the unsupported notice, preserves
  its old report as stale source and downloads the exact **10,260 characters**.
  Restoring correct code returns 11/11. Temporary verification downloads are
  removed. Private worker continuation is checked separately with the pinned core.

These are existing synthetic teaching tasks, not dataset/paper benchmark claims.
Observed localhost BPR phases are preparation 1.80 s, calibration 1.17 s and
complete learner tests 0.16 s; they are not internet cold-start or Oracle hardware
measurements. Source-local probe checks do not replace any maintained lesson test.

## Runtime identity and release boundary

The final export adds one shared autodiff runtime, 67 manifests and 67 details
(**135 immutable assets**), retaining all **2,258 prior data files**. Catalog and
index change only browser/detail references. Schedules is unchanged, as are all
68 modules, 53 paper records, 538 lesson checks and 287 maintained faults. Only
uncommitted intermediate assets from this work are removed; committed immutable
assets remain intact.

The complete frontend checks and standalone/hosted builds pass. Initial graphs
remain **401,047 raw / 129,981 gzip bytes** standalone and **406,423 raw / 131,823
gzip bytes** hosted, within the original 400 KiB / 130 KiB limits. The operator
change lives in the shared Python runtime asset and adds no initial JavaScript.
The browser warning/error logs are empty; the static server records **179 GETs,
zero API calls/POSTs and zero HTTP errors**. The final BPR Lab displays 120 updates,
loss 0.69427 to 0.034932, pair accuracy and training Recall@2 both 1.0. Agent-owned
tabs, static server and verification downloads are closed; the build snapshot is
removed after committing and ignored evidence/screenshots are retained.

The verified source snapshot excludes concurrent account, UI, publication-policy
and deployment work. No account data or implementation moves into the public
repository. No push or production application/image deployment is performed;
the approved Nginx-only release is unchanged.

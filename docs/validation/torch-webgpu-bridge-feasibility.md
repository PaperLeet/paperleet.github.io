# WebGPU bridge and sandbox feasibility

This is a capability experiment, not a supported Torch device or performance
claim. The supported runtime continues to execute through NumPy on CPU/WASM.

## Measured execution contexts

The probe used Pyodide 0.29.4 and unmodified headless Chrome 155.0.8059.82 on an
Apple Metal adapter. No browser feature flags or permission changes were added.
The local loopback fixture supplied wildcard CORS for public runtime assets.
It compared a regular Worker with a blob Worker created inside a
`sandbox="allow-scripts"` iframe, using the current judge's restricted CSP pattern.
This reproduces the relevant isolation configuration, not a production-site
deployment test. Record fresh observations for other browsers and hosting setups.

| Observation | Regular Worker | Opaque-origin sandbox Worker |
| --- | --- | --- |
| Secure context | yes | no |
| JSPI `Suspending` / `promising` exposed | yes | yes |
| Pending Promise through synchronous `runPython` | rejected | rejected |
| Pending Promise through `runPythonAsync` | returns 37 | returns 37 |
| `navigator.gpu` exposed | yes | no |
| GPU adapter | Apple / metal-3, non-fallback | unavailable |
| Python → WGSL dispatch → mapped readback → Python | returns 42 | not attempted |

Each context ran the reusable probe twice in the same interpreter with identical
observations and no leftover probe globals. A real timer-delayed Promise prevents
mistaking an immediately available value for working suspension.

The synchronous entry reported: “Cannot stack switch because the Python
entrypoint was a synchronous function.” The asynchronous entry supported
`pyodide.ffi.run_sync`. This establishes that the bridge can suspend on this
runtime; it does not establish tensor, autograd or model compatibility.

## Consequences for implementation

Replacing only the matmul shader cannot enable the current judge backend.
There are two separate constraints: GPU exposure in the isolated execution
context, and the synchronous entry used for calibration and learner execution.
Keep learner isolation while evaluating a bounded GPU service in a separate
secure host Worker. Such a service would need explicit buffer ownership,
resource limits, cancellation and validated operations. Switching to an
asynchronous Python entry also needs complete judge compatibility verification.

Only after those boundaries work should a backend introduce resident tensor
storage, explicit CPU transfers, dtype/operator support and analytical
gradients. Compare complete workloads including conversion, upload, dispatch,
readback and transfer boundaries before choosing dispatch thresholds.

## Reproduction and evidence

The reusable developer probe is
[`webgpu_bridge.mjs`](../../packages/torchwasm/benchmarks/webgpu_bridge.mjs).
Call `probeWebGpuBridge(py)` with an initialized Pyodide instance inside each
execution context. It records unavailable capabilities rather than enabling a
backend and releases its temporary Python globals and GPU resources.

The [raw observation](../../packages/torchwasm/benchmarks/results/webgpu-bridge-capabilities.json)
records browser/runtime versions, source hashes, requests from the opaque
origin, Python diagnostics and both context results. Its scope is one scalar
GPU calculation, without speed, memory, gradient or training claims.

# Backend selection review for the 0.78 candidate

Keep the supported Tensor backend on CPU/NumPy and NumPy/WASM while the 0.78
candidate finishes qualification. Continue developing the separate WebGPU
experiment toward an explicit opt-in backend. The available measurements do
not justify automatically sending ordinary Torch operations to the GPU.

This is a decision from existing project evidence, not a new GPU measurement.
It preserves the long-term GPU investigation and the goal of a standalone
Torch-compatible library. It changes no device policy, shader, sandbox, default
quota or production runtime.

## Evidence and its limits

The [iteration 74 measurements](torch-runtime-iteration-74.md) include Python/JS
conversion, RPC, dispatch, readback and release. The separate resident profile
excludes initial input upload. Large 256/512 dense products benefited on the
tested Apple Metal adapter; small attention-score, two-product chain and small
explicit matrix-VJP programs were slower than their NumPy/WASM counterparts.
Those are finite float32 programs, not complete attention or automatic training.

[Iteration 75](torch-runtime-iteration-75.md) reduced GPU overhead by submitting
several products/readbacks in one bounded transaction. Its small chain and VJP
programs still took about 4.6x and 2.4x the NumPy reference time in the resident
profile. The six-process data also retain short batches and variable dense512
A/A controls, so these observations do not establish a routing threshold.

Crucially, those GPU comparisons predate the
[iteration 76 WASM Tensor matmul optimization](torch-runtime-iteration-76.md)
and later batched, empty-product and affine work. The old NumPy reference does
not exercise today's public Tensor dispatch. Ratios from those different runs
must not be combined into a synthetic current GPU/CPU comparison. A fresh
comparison must include current Tensor forward and gradients on the same
operands, alongside raw NumPy and the GPU program.

[Iteration 78](torch-runtime-iteration-78.md) strengthens primitive/program
rollback, acknowledged cancellation, ownership and real GPU recovery. Its
finite 2D oracles and explicit analytical products remain narrower than the
current CPU/WASM Tensor and autograd surface.

## Runtime integration that remains necessary

| Boundary | Current supported implementation | GPU work still needed |
| --- | --- | --- |
| Tensor values | NumPy arrays, including validated strided/broadcast cases | Resident handle storage with shape/dtype/ownership metadata and explicit transfers |
| Gradients | Validated reverse/forward Tensor rules and supported higher derivatives | Backend-aware seeds, accumulation, hooks, saved values and release; explicit VJP products alone do not provide these |
| Python entry | Existing synchronous judge execution | Qualify asynchronous entry for calibration, learner execution, progress, deadlines and cancellation |
| Sandboxed access | Opaque-origin Python Worker | Preserve that isolation and connect a bounded secure host service; the measured sandbox Worker does not expose GPU access |
| Precision and values | Existing validated dtype, special-value and empty rules | Independently qualify each GPU dtype/operator/value family; retain explicit unsupported behavior or validated CPU fallback |

The [measured bridge](torch-webgpu-bridge-feasibility.md) distinguishes direct
host access from the opaque-origin Worker. The pinned
[Pyodide 0.29.4 API](https://pyodide.org/en/0.29.4/usage/api/python-api/ffi.html#pyodide.ffi.run_sync)
requires JS Promise integration and a compatible asynchronous Python entry for
`run_sync`, and marks that API experimental. GPU readback itself is asynchronous:
[`mapAsync`](https://developer.mozilla.org/en-US/docs/Web/API/GPUBuffer/mapAsync)
resolves when buffer contents can be accessed and mapped buffers cannot
participate in GPU commands. These constraints agree with the local probe;
switching only a matmul shader does not satisfy the judge contract.

Cancellation also needs its existing distinction between discarding a result
and preempting submitted GPU work. Device destruction is not evidence of
preemption: [the device API](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/destroy)
allows already enqueued commands to execute before destruction. Keep the
prototype's drain/rollback and invalidation behavior while qualifying the
application-level deadline and ownership rules.

The source currently exposes the CPU/WASM policy in `_device.py`, stores NumPy
values in `Tensor._data`, and uses NumPy-array pullbacks in `_graph.py` and the
operator modules. Introducing resident storage therefore requires actual
storage and derivative dispatch work. Implicit readback at each existing NumPy
call would undermine residency and needs separate measurement rather than an
assumed speedup. The startup alias can continue serving ordinary `import torch`;
backend selection belongs in runtime setup, without rewriting learner imports.

## Next GPU experiment after the current timing jobs

Use the unchanged supported 0.78 source as the CPU/WASM reference. Compare
representative tiny, 128-scale, 256-square and 512-square finite float32 products,
then whole forward-plus-two-VJP programs. Keep separate roundtrip and resident
profiles. Include current Tensor dispatch, raw NumPy, sequential GPU and grouped
GPU paths; share operands and gradient seeds, verify against the native/float64
controls, and retain all fresh-process samples and A/A controls. Report import,
initial upload, repeated call and final readback boundaries explicitly.

The existing work quota also affects the comparison. A 512-square forward and
its two matrix gradients require 402,653,184 multiply-accumulate terms in total,
exceeding the default 268,435,456 work limit for one program. Partition that
workflow and include the resulting dispatch/ownership costs; do not increase
the default quota to make the comparison look like one transaction. The
[service contract](../../packages/stridetorch/benchmarks/webgpu_rpc/README.md)
documents the remaining memory, handle, matrix and lease limits.

After numerical and whole-program cost evidence supports a target workload,
qualify the storage/autograd interface and the actual judge's asynchronous
execution before exposing an opt-in device/backend. Automatic routing remains
a later decision requiring a stable benefit across relevant shapes, layouts,
dtype/fallback boundaries and complete workloads. No new GPU job runs alongside
the current CPU/WASM latency measurements.

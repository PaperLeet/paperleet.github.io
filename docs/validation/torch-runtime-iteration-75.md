# Torch runtime iteration 75: bounded WebGPU matrix programs

The stable library remains **0.73.0**, with its CPU/NumPy/WASM backend. This
iteration extends the experimental host-Worker matrix service; no public torch
GPU device, automatic autograd, import alias or production judge is enabled.

`MatrixProgram` submits a bounded DAG of fixed matmul operations in one request.
Inputs may be copied float32 NumPy arrays or same-session resident matrices.
One command submission contains the compute passes and final packed readback.
Requested outputs may be read, retained, or both; intermediate buffers are
released and retained outputs close through an atomic `release_many` request.
The [portable sources and examples](../../packages/stridetorch/benchmarks/webgpu_rpc/README.md)
have no PaperLeet imports.

## Validation and failure handling

The host validates every reference/shape/flag, aggregate compute work, published
handle count and peak byte requirement before GPU allocation. Default limits
remain 32 MiB tracked GPU bytes, 24 resident handles, eight pending requests,
4096 requests and a 120-second lease. Programs accept at most eight inputs,
eight matmul nodes and eight distinct computed outputs. The 268,435,456 MAC
limit applies to all nodes together; uploads, temporary values, per-node uniforms
and packed readback count toward the same byte budget.

An early review found that queue writes may precede a later encoding/mapping
failure. Program rollback now drains pending work before freeing tracked
resources; a failed drain invalidates the device. Preflight rejections preserve
existing resident handles. Cancellation skips queued requests or discards all
new retained outputs after completion, rather than preempting GPU kernels.
The original source and its narrower legacy evidence are retained. A first test
expectation incorrectly required a second queue fence after successful mapping;
its failure/source are also retained, with the corrected map-completion check.

* 36 original native Torch 2.6.0 and independent float64 matrix/VJP fixtures,
  each executed with ephemeral uploads, retained outputs and resident inputs:
  **108 program executions**, not 108 distinct native fixtures. All four
  transpose combinations and nine irregular/tile-boundary shapes retain
  `rtol=1e-4`, `atol=5e-4`. Every program execution adds one submission.
* Legacy 36 native fixtures and 13 real boundary rows still pass. Four additional
  real program checks cover aggregate work/byte rejection, two-output
  cancellation recovery and separate packed/read/retained snapshots for three
  large outputs. Both real cancellation attempts in this run were acknowledged
  and their new retained outputs reclaimed.
* 40 actual planner/engine/RPC fault-injection tests using a fake GPU surface,
  plus nine legacy protocol tests. These test allocation/encoding/submission/
  mapping/read/scope failures, drain failure, full-budget preflight, atomic
  release and multi-output cancellation; they do not execute a shader.
* Four additional real GPU lifecycle checks retain the iteration-74 scope:
  startup disposal, resident-handle invalidation, short trusted lease recovery
  and explicit device destruction to exercise device-lost handling. They do not
  simulate hardware crashes or run their clients in the opaque Python Worker.
* The repository copies pass **66 namespace/feature tests** and
  **9 built-wheel/ownership/install tests**. The stable runtime's
  61 resources and live exports are unchanged. The 42,477/full-catalog/frontend
  gates from iteration 73 are not rerun or presented as new program checks.

The real numerical path remains an opaque-origin `sandbox="allow-scripts"`
iframe/Worker, without direct GPU access, connected to an owned secure host
Worker. Unmodified headless Chrome 155.0.8059.82/Apple Metal3 uses no extra flags.
All successful sessions finish with zero tracked allocations/handles; this is
API accounting/destruction evidence, not physical VRAM reclaim. The actual
nested ndarray conversion produces `Float32Array`, no PyProxy, and permits the
sample Python source allocation to be collected; this is one schema observation,
not a general heap/RSS benchmark. A Node ESM loader inspection failed before
conversion, so the observation was made in the real Chrome fixture. Its compact
diagnostic is archived; the full minified-line log stays local.

## Sequential requests versus one program

Both GPU sides use the **same candidate engine** and unchanged WGSL/legacy
primitive bodies. A and A′ are identical sequential callables; B builds/runs
one program for the same mathematical outputs. NumPy/WASM is a separate context
baseline. Roundtrip profiles include input conversion/upload and all output
readback/release. Resident profiles exclude one-time input uploads on both GPU
sides. Program construction and Python/RPC costs remain timed. Adapter/pipeline/
interpreter initialization is excluded. These are explicit matrix programs,
not public torch/autograd/native-CPU or complete attention/training comparisons.

Six workflows × two transfer profiles × six fresh Chrome/Pyodide/device
processes give **72 children**, each with eight balanced batches of four paths.
The initial 72-child run retained 503/2304 batches below20ms. A second run uses
40ms calibration and longer iteration caps: **3/2304** batches remain below20ms,
with minimum sequential/A′/program/NumPy batches
18.9/19.8/19.8/38.6ms.
For this long sampling only, trusted host code raises the **request count** to
32768. Default4096 and GPU byte/work/handle/pending/lease limits are unchanged;
untrusted clients cannot change limits. Actual served runtime assets have six
consistent hashes across all measured children; NumPy is 2.2.5/Pyodide0.29.4.

Ratios are medians of six per-process ratios. Descriptive microsecond medians
are aggregated separately and need not reproduce those ratios.

| Program/profile | A′/A | B/A | B/A′ | B/NumPy | A → B μs |
| --- | --- | --- | --- | --- | --- |
| paperleet_attention_score_head_roundtrip | 0.9910 | 0.3207 | 0.3225 | 641.5337 | 1606.25 → 525.98 |
| paperleet_attention_score_head_resident | 0.9907 | 0.5350 | 0.5415 | 614.5399 | 902.34 → 485.74 |
| attention_score_128_roundtrip | 0.9727 | 0.4264 | 0.4421 | 1.7046 | 1614.06 → 714.84 |
| attention_score_128_resident | 0.9495 | 0.5890 | 0.6090 | 1.4061 | 990.63 → 594.34 |
| dense_256_roundtrip | 0.9854 | 0.6307 | 0.6406 | 0.1420 | 2451.56 → 1530.47 |
| dense_256_resident | 0.9899 | 0.7298 | 0.7374 | 0.1154 | 1277.34 → 1137.11 |
| dense_512_roundtrip | 1.0084 | 0.8667 | 0.8765 | 0.0822 | 8393.75 → 7440.62 |
| dense_512_resident | 0.9112 | 0.8402 | 0.8987 | 0.0385 | 3993.75 → 3354.69 |
| resident_mlp_chain_roundtrip | 1.0048 | 0.3180 | 0.3159 | 5.5346 | 2244.53 → 713.87 |
| resident_mlp_chain_resident | 1.0338 | 0.4019 | 0.3874 | 4.6026 | 1491.41 → 601.17 |
| linear_vjp_roundtrip | 0.9913 | 0.2425 | 0.2424 | 2.7665 | 3854.69 → 944.14 |
| linear_vjp_resident | 0.9877 | 0.2595 | 0.2617 | 2.4044 | 2967.19 → 766.41 |

The matrix VJP includes the output and two explicit analytical matrix products;
the chain contains two products without activation. Relative to sequential GPU
calls, these multi-operation profiles show clear benefits against both A/A
denominators. They remain slower than NumPy for these small fixtures. Dense512
gains are smaller/variable: resident A/A median0.9112 and individual A/A range
0.7839–1.1136 across profiles caution against a precise universal gain. The first
wave is preserved, and no automatic routing threshold is selected. Default
CPU/WASM execution remains appropriate while GPU storage/autograd and real
judge async/budget/cancellation integration are developed.

## Reproduction and scope

The [WebGPU encoder documentation](https://developer.mozilla.org/en-US/docs/Web/API/GPUCommandEncoder)
describes collecting multiple commands into a submitted command buffer.
The [buffer-copy requirements](https://developer.mozilla.org/en-US/docs/Web/API/GPUCommandEncoder/copyBufferToBuffer)
apply to the packed readback offsets/sizes. Cancellation and device destruction
do not establish preemption. Transport queues and browser/driver resources lack
a hard tenant memory/DoS guarantee. Float64/complex/special values,
batched/broadcast/empty tensors, automatic torch AD and a supported GPU backend
remain future work; large graphs may need partitioning to respect budgets.

Namespace/wheel checks and original qualification executed on
`b28a2fd31f681d108c85252b151069e0ba044bc4`. Main then advanced through 28 frontend/docs
files to `9dfcb2c956492653d648ba4e1a6ccfe9f21bd775`. Those updates were retained with a fast-forward;
runtime/Python/test/package/dependency/judge inputs and measured runtime assets
are unchanged. This is explicit source-equivalence, not a claim of re-execution
on the newer SHA or updated frontend testing for a benchmark-only change.

Machine report: `packages/stridetorch/benchmarks/results/iteration-75.json`.
Raw harness `iteration-75-harness.tar.xz`: 1,810,896 bytes,
SHA-256 `523eefc2da2f4034449fc3529e869f5c814d1cb1dcf185c0a7948c1103e781c9`; 409 logical files,
370 mode/ownership-compatible payloads,
39 hardlinks; every logical extracted payload verified.
It preserves prototypes, test failures, both timing waves and receipts, with
historical absolute paths/dependency setup required. Remote integration is
verified separately. No CI, tag, package release or deployment is requested.

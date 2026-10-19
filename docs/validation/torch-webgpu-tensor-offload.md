# Explicit Tensor matmul offload prototype

`MatmulOffload` implements host GPU forward multiplication and actual
first-order autograd pullbacks for the existing CPU Tensor graph. Standard
`autograd.grad`, `backward`, hooks, retained graphs and an SGD optimizer consume
its results. The implementation lives in
[torch_bridge.py](../../packages/stridetorch/benchmarks/webgpu_rpc/torch_bridge.py),
with [usage and setup](../../packages/stridetorch/benchmarks/webgpu_rpc/README.md).
It is an explicit experimental helper, depends on private 0.83 interfaces and
does not change the stable package API, ordinary `@` dispatch or Tensor device.

## Execution and state contract

Finite float32 2D operands within conservative numerical bounds use the existing
bounded MatrixProgram service. Forward and active pullbacks are GPU programs;
only requested gradients are computed. The helper splits programs at the
existing aggregate MAC limit, including the 1024-by-512-by-512 numerical case.
Outputs and gradients return to CPU storage. No resident GPU handles survive
the call, and no arbitrary shader code or larger quota is introduced.

The helper snapshots input values, identities, shapes, dtype, activity, graph
metadata and storage versions before forward submission. It checks the saved
state and value bits after each program before creating a public result. The
backward route checks required original versions, reads current shared factors,
then snapshots them and the seed before waiting. Bitwise checks include NumPy
writes that bypass Tensor counters and signed-zero changes. Obsolete forward
copies are released; both pullbacks reuse one owned seed copy.

Reading current factors at backward entry preserves native CPU matmul's
behavior when a NumPy shared-storage write occurs between calls. Changes
during a hidden wait are rejected, including tracked writes through aliases,
untracked NumPy writes, input-activity changes, global AD-mode changes and seed
mutation. GPU transport failures propagate without a silent CPU retry. The
module gate is reset on failure and rejects reentrant offload calls; it does
not serialize all possible Python callbacks.

`no_grad`, forward/higher-order AD, other dtypes, vectors, batches, empty
matrices and exceptional ranges use the existing CPU operation without a hidden
wait. Backward explicitly invoked with recording disabled also uses CPU
pullbacks. The no-grad callback probe confirms its queued callback runs after
the caller's temporary context exits. A general task-isolated GPU execution
model remains future work.

## Numerical, autograd and lifecycle evidence

- Native NumPy 2.5.4 / Torch 2.14.1 and NumPy 1.26.4 / Torch 2.6 each pass 86
  prototype tests. Repository-local copies pass the same 86 tests.
- Actual Chrome 155 / Pyodide 0.29.4 / NumPy 2.2.5 passes 51 independent
  value/gradient cases against both native Torch references, using frozen
  `rtol=1e-5, atol=1e-6`. Cases include C/F/sliced inputs, four activity modes,
  duplicate input edges, and sizes through 1024-by-512-by-512.
- Five nonlinear SGD steps with momentum, weight decay, a transposed nonleaf
  weight and a bias match the native references. This checks training
  numerics, not whole-model training throughput.
- Nineteen actual-browser controls cover tracked/raw mutations, aliases,
  activity and grad-mode changes, reentry, seed changes, hooks/retain-grad,
  retained-graph consumption, fresh gradient buffers, CPU fallback and NumPy
  changes between calls. Caller gradient buffers and input ownership are
  checked. All GPU handles, bytes and allocations return to zero; host disposal
  confirms a closed service. The numerical probe uses 124 requests and has a
  tracked buffer high-water mark of 7,340,096 bytes.
- Unchanged portable service checks pass nine protocol, 40 program, 39
  primitive and four cancellation/drain cases. These inject a fake engine;
  the browser checks above use a real Apple `metal-3` GPU adapter.

The original native invocation remains failed: 74 passed and two failed due to
test setup (unsupported direct nonleaf mutation syntax and a wrong handwritten
HVP expectation). A fresh test source uses a detached output alias, an
independent native HVP result and asserted physical source strides. The first
81-test and 51-case GPU gates passed but did not cover raw NumPy writes.
Code inspection then found that boundary; all five added regression cases fail
on the original prototype and pass on the corrected implementation. Original
sources and results remain separate in the archive.

## Matched whole-call measurements

Three fresh Chrome processes measure the canonical 0.83 CPU runtime and the
qualified offload source. Both providers run in Pyodide/NumPy WASM and perform
forward plus actual `autograd.grad`, returning three independent owned CPU
arrays. Native Torch supplies numerical controls. Input CPU Tensors are reused
in both providers; GPU snapshots, bit checks, uploads, readback, graph work and
returned-array copies remain inside each measured call. No GPU input residency
is cached. Startup and initial shader preparation precede measurement.

| Matrix product | CPU median ms | GPU median ms | Median paired GPU/CPU ratio |
| --- | ---: | ---: | ---: |
| 37 × 53 × 29 | 0.0910 | 1.5672 | 17.2621 |
| 64 × 64 × 64 | 0.3488 | 1.7453 | 5.0030 |
| 256 × 256 × 256 | 10.6417 | 5.2607 | 0.4947 |
| 512 × 512 × 512 | 79.4500 | 16.6750 | 0.2119 |

On this adapter the two larger workloads benefit by about 2 and 4.7 times;
small workloads lose to CPU. The paired ratio ranges across the three processes
are 0.4915–0.5254 and 0.2080–0.2129 for the two larger cases. These session-local
paired comparisons do not establish a universal size threshold or a speedup
relative to an earlier version's separately measured absolute times.

The frozen plan keeps ordinary GC enabled, collects explicitly outside each
timed batch, uses twelve balanced A0/A1/G orders and retains every sample.
Qualification requires both aggregate estimators to pass 5% A/A controls,
every individual A/A estimate within 10%, and all batches at least 30 ms.
All twelve process/shape contexts satisfy these gates, with no individual
flags. Observed minimum batches range from 39.1 to 60.2 ms across the four
shapes. The raw per-row console ledger matches each final primary report.
Default 32 MiB, request, MAC, handle and session limits remain unchanged, and
every process closes with zero tracked allocations.

## Integration and remaining work

Canonical 0.83's 66 resources, all 93 SDK sources, package metadata and 183
active export identities are unchanged. This change adds benchmark/developer
code, tests, usage documentation and evidence; it enables no production GPU
backend. Stable independent-package API design, resident Tensor storage,
operation chaining, wider dtype/operation coverage, task isolation and tests on
other browser/device combinations remain necessary before default integration.

[Machine-readable qualification](../../packages/stridetorch/benchmarks/results/webgpu-tensor-offload.json)
and its [verified raw archive](../../packages/stridetorch/benchmarks/results/webgpu-tensor-offload-harness.tar.gz)
preserve original and corrected prototypes, native controls, actual browser
fixtures, frozen plans, every timing sample and replay dependencies. The archive
contains 374 files, 142 unique payloads and SHA-256
`f3dc8c25b1d4c727be9560ea01c258177c8a5fc04696e33fc5eedf2c38a4bed5`.
The qualified helper's SHA-256 is
`f7cb57084191430d907f7502b6249413d98c933a7dd4fb91bcf6fa556818c2e6`.
No CI dispatch, tag, publication or deployment is requested. The runtime
improvement goal remains active.

Main adopted the explicit helper and evidence at
`748985d3b3553648927e279b2a873f0fc1dbafaa`. The
[remote adoption receipt](../../packages/stridetorch/benchmarks/results/webgpu-tensor-offload-main-adoption.json)
verifies the helper, tests and all nine benchmark/documentation/evidence paths
from that commit, together with 66 unchanged runtime resources, 183 export
identities, seven release-policy files and all 206 qualified package inputs.
The delta from the then-current remote main contains no application changes.
Existing frontend updates on remote main were merged, with five targeted
compatibility checks passing; no new full application build is claimed for this
benchmark-only integration. The receipt observes zero workflow runs for the
adopted SHA and Pages remains on `gh-pages` at `/`.

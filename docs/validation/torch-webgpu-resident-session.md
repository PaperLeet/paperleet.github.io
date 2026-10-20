# Bounded resident Tensor offload sessions

`ResidentMatmulOffload` adds explicit GPU buffer reuse to the qualified Tensor
matmul helper. Inputs, forward intermediates and a pullback immediately consumed
by another GPU node can remain resident. Public values and gradients are CPU
Tensors. Every requested matrix computation still executes; this is buffer
reuse, not numerical-result memoization. The implementation and setup are in
[the developer prototype](../../packages/torchwasm/benchmarks/webgpu_rpc/README.md).

## Ownership and protocol changes

The default cache owns at most eight matrices and 8 MiB of logical payload in
private CPU shadows and corresponding GPU buffers. These bounds exclude
per-call snapshots, readback and other temporary allocations. Cache options can
only reduce them. All original host limits, including the 32 MiB aggregate GPU
budget, 24 handles and 268,435,456 MACs per program, remain unchanged.

The existing program schema can now export newly uploaded inputs with
`read=false, retain=true`. Existing resident inputs cannot be exported as second
owners. Input exports and computed outputs share the eight-result and handle
limits. `MatrixProgram.run` adds `retain_inputs` and per-computed-output
`retain_outputs`; its defaults retain their previous behavior.
`ProgramResult.input_matrices` owns the requested fresh input handles, and its
context manager releases both those handles and retained computed outputs.

The session plans admission and evictions before dispatch, pins the current
program's input handles, releases victims together and retains new uploads in
the computation request. Inputs that cannot fit are submitted inline. A small
signature filters cache candidates; full value bits still determine reuse.
Tests cover distinct interiors with equal sampled positions. The shader and
GPU service execution/rollback code are unchanged; planner validation, client
ownership and developer wrappers receive the extension.

Each call retains the original input/seed version, identity and bit checks.
Failures roll back newly owned leases. If cleanup fails, ownership records are
kept and the session rejects reuse until explicit cleanup. A saved backward
cannot steal an outer call's reentry gate. `clear()` releases cached buffers
without closing the session. Successful `close()` releases ownership and is
idempotent; new forwards then fail while existing CPU Tensor graphs can finish
through CPU pullbacks. Close and clear are explicit RPC suspension points.
The prototype does not establish general Python task isolation.

## Functional and lifetime evidence

- Both NumPy 2.5.4 / native Torch 2.14.1 and NumPy 1.26.4 / native Torch 2.6
  pass 123 tests. The repository-local integration passes the same 123 tests.
- Actual Chrome 155 / Pyodide 0.29.4 / NumPy 2.2.5 passes the existing 51
  independent native value/gradient contexts and 31 state, ownership and reuse
  controls. These include C/F/sliced two-node GPU graphs and five steps of
  nonlinear two-node SGD training against both native references.
- Controls cover tracked and raw NumPy writes, signed zero, activity/mode and
  seed changes, hooks, retained graphs, close-after-forward CPU continuation,
  reentrant saved backward, tiny caches, exact signature collisions and direct
  fresh-input retention. Numerical tolerances remain `rtol=1e-5, atol=1e-6`.
- Portable checks pass 63 program, nine protocol, 39 primitive and four
  cancellation/drain cases. Program checks include all original cases plus
  retained-input rollback at 18 injection points, handle-budget preflight,
  existing-handle alias refusal and cancellation of all new input/output
  handles. These checks use an injected GPU surface; the browser checks above
  use the real Apple `metal-3` adapter.
- Numerical probes and timing sessions all end with zero tracked allocations,
  handles and bytes, followed by confirmed host disposal. Cache shadow/handle
  counts are also checked against live service statistics.

## Direct transfer trace

An untimed real-browser trace intercepts actual MatrixRpc program descriptors
and returned typed-array lengths for a 37×53×29×17 two-node graph:

| Provider and phase | Program RPCs | Compute nodes | Input-array upload bytes | Readback bytes |
| --- | ---: | ---: | ---: | ---: |
| Stateless, cold | 4 | 6 | 47,320 | 27,064 |
| Stateless, repeated | 4 | 6 | 47,320 | 27,064 |
| Resident, cold | 4 | 6 | 18,480 | 27,064 |
| Resident, repeated | 4 | 6 | 0 | 27,064 |

The cold resident graph uploads the original three operands and final seed;
its forward intermediate and upstream gradient reuse buffers. Repeated calls
still execute all six kernels. Upload figures cover matrix-array payloads,
excluding protocol headers and uniforms. CPU results and the current graph
engine still require readback; this is not a zero-transfer Tensor backend.

## Paired performance campaign

Six fresh Chrome processes cover 256 and 512 square matrices, three processes
per size. Each measures a single-node repeated-input flow, a two-node repeated
graph and a flow that updates the right operand by a fixed small step after
each forward/backward call. The changing case stresses invalidation and is
verified against independent native affine references; it is not a model
training throughput measurement.

The two controls run the previous stateless Tensor GPU implementation. All
providers share the updated transport and adapter. Every call includes actual
autograd, snapshots, bit checks, cache logic, per-call evictions, uploads and
owned output copies. Single-node flows return three owned arrays; chained
flows return five. Session setup and final close are outside timing and are
checked separately. Ordinary GC stays enabled, with explicit collection outside
each batch. All raw samples and per-row durable ledgers are retained.

| Size and flow | Stateless median ms | Resident median ms | Paired resident/stateless ratio |
| --- | ---: | ---: | ---: |
| 256, repeated single | 4.3700 | 3.4208 | 0.7842 |
| 256, repeated chain | 8.8571 | 6.6389 | 0.7650 |
| 256, changing right operand | 4.5679 | 3.9250 | 0.8343 |
| 512, repeated single | 15.8750 | 12.3200 | 0.7778 |
| 512, repeated chain | 30.7250 | 24.5167 | 0.7835 |
| 512, changing right operand | 15.8875 | 13.2300 | 0.8355 |

Across the two frozen estimators the observed reductions are approximately
14–25% relative to stateless GPU calls in these sessions. These are paired,
adapter-specific results and do not establish a universal routing threshold or
a speedup over separately measured absolute times from an earlier campaign.

The preplanned rule requires both three-process median A/A estimates within
5%, every individual A/A estimate within 10%, and all batches at least 30 ms.
All six contexts pass. Two individual unpaired A/A estimates exceed the 5%
individual band: 0.9350 for the 256 chain in cycle 1 and 1.0561 for the 512 chain
in cycle 2. They remain flagged, not relabeled. Their aggregate controls pass
the rule frozen before execution. Minimum batch times range from 38.7 to
57.4 ms across the six contexts.

## Preserved first candidate and scope

The first resident implementation uploads inputs with separate RPCs and uses
an unfiltered cache scan. Its full six-process campaign is preserved. Repeated
flows benefit, but continuously updated inputs cost about 14% at size 256 and
4% at size 512 under the two estimators; it is excluded. Batched input retention,
batched eviction, selective output retention and exact-checked signature
filtering address that measured overhead. No favorable individual context was
rerun or selected to replace a failed measurement.

The canonical 0.83 runtime's 66 resources, 93 SDK sources, all 206 package inputs
and 183 active export identities remain unchanged. This change affects only
developer GPU code, tests, documentation and evidence. Stable independent
package APIs, feature negotiation, wider operation/dtype/device coverage,
task isolation and resident Tensor storage remain future work. No production
GPU dispatch, CI, tag, publication or deployment is enabled.

[Machine-readable qualification](../../packages/torchwasm/benchmarks/results/resident-session.json)
and the [verified raw archive](../../packages/torchwasm/benchmarks/results/resident-session-harness.tar.gz)
contain both candidates, native references, numerical/ownership probes, both
six-process campaigns, transfer traces and replay dependencies. The archive
has 504 files, 172 unique payloads and SHA-256
`c423effe0ff2d72f0178ba223aca8948013da3c5be922e7ba85f3324ed09177b`.
The runtime improvement goal remains active.

## Main adoption

Commit `4880ee6476f2c606cc79d83a33a9bd998b814d6a` adopts the qualified
experimental resident session on main. The remote commit's ten feature files,
66 runtime resources, 183 active exports, seven release policies and all 206
package inputs were verified directly from its Git tree. Eight checks passed
after integrating the then-current frontend changes. The default CPU/WASM
runtime remains 0.83.0; GPU dispatch remains opt-in developer code.

The guarded ordinary push created no workflow run for this commit. Pages still
uses the legacy `gh-pages` branch. No tag, publication or deployment was made.
The [main adoption receipt](../../packages/torchwasm/benchmarks/results/resident-session-main-adoption.json)
and its [raw evidence](../../packages/torchwasm/benchmarks/results/resident-session-main-adoption-harness.tar.gz)
record the remote verification and push checks, including the initial integration
hold for a separately reviewed frontend package-script addition.

# Lightweight torch 0.9 validation

This milestone makes reverse graphs release saved numerical state after a default
backward pass, while retaining lightweight history and non-leaf identity. Explicit
`retain_graph=True` keeps the saved state for another pass. The standalone runtime
remains NumPy-only and runs unchanged torch-importing scripts through the existing
launcher or host aliases.

## Behavior

- `Tensor.backward`, `torch.autograd.backward` and `torch.autograd.grad` support
  flat multiple roots and matching seeds. Duplicate roots sum their seeds;
  duplicate differentiation inputs preserve their requested result positions.
  Exact dictionary inputs produce named gradient dictionaries.
- Requested-input pruning avoids unrelated ancestors. `grad` returns derivatives
  without accumulating leaf `.grad`; retained non-leaf gradients still accumulate.
  `backward(inputs=...)` accumulates only requested inputs, including retained
  non-leaves. It executes a selected non-leaf's gradient function, whereas `grad`
  can stop at that node. Tests cover the observable difference after an ancestor
  has already released saved state, including partial retained-gradient updates.
- `retain_grad()` enables non-leaf `.grad` accumulation and `retains_grad` reports
  it. Leaf retention is a no-op. Frozen tensors reject retention. Existing
  gradient objects keep their identities during accumulation.
- Saved callbacks, captured arrays and version checks are released after each
  successfully executed node unless retention was requested. History preserves
  parent relationships, shape, dtype and release status. A second pass requiring
  freed state raises RuntimeError; the output remains non-leaf.
- Retention is operator-specific. Addition, basic slicing, shapes, sum/mean,
  concatenation, constant-spacing trapezoid, integer-count repeat-interleave,
  bicubic resize and global average pooling have repeatable metadata-only rules.
  Scalar multiplication/division, advanced indexing, regular area resize and
  regular adaptive average pooling retain native saved-state requirements.
  Remainder is repeatable when only its dividend needs gradients.
- Saved state remains guarded against in-place mutation. Only executed branches
  check their saved state. Unsupported mutation still reports the established
  typed fallback rather than silently computing from changed values.
- Unused inputs support `allow_unused` and `materialize_grads`. Materialized
  zeros are frozen under the validated Torch 2.14 `create_graph=False` contract.
  Invalid seeds, shapes and differentiation flags fail before consuming a valid
  graph. General higher-order graphs, batched seeds, GradientEdge inputs and
  backward inside a forward AD transform remain unsupported.
- Sum/mean validate scalar, empty-list/tuple and duplicate/out-of-range dimensions;
  mean accepts a floating/complex dtype. Differentiable integer-dtype sum raises
  RuntimeError; it works under no-grad. Float32 mean gradients preserve dtype
  under both NumPy 1.26 and 2.x.
- Scalar int32/int64 Tensor indexing snapshots a basic integer index and preserves
  zero-dimensional view storage and shared version counters. Constant-spacing
  trapezoid uses a direct analytical rule and matches covered native NaN, infinity,
  empty, single-sample and subnormal boundaries.

Native-reference behavior targets installed Torch 2.14.1 and the public
[grad](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad.html) and
[backward](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.backward.html)
interfaces. This is a bounded autograd implementation; arbitrary native graph
introspection, Tensor/backward/global hooks and general higher-order reverse
execution are outside this milestone. Forward AD and reverse-over-forward remain
covered by existing numerical tests. Two older adapter tests now explicitly keep
shared graphs for their second pass, matching the native side of those tests.
No lesson solution or mutant was changed to accommodate the implementation.

## Memory and execution

Intermediate Tensor wrappers no longer form the reverse history. Non-leaves use
small graph nodes; leaves are referenced directly, avoiding an extra leaf node or
self-cycle. Only explicit non-leaf retention allocates a weak owner reference.
Metadata-only VJPs capture shapes/dtypes instead of input arrays. The reverse
engine removes consumed temporary gradients during traversal.

The memory fixture allocates a 64×128 input before tracing, applies sixteen
add/tanh stages and a squared mean loss, drops user-held intermediate variables
and keeps the scalar loss alive after backward. Both versions produce identical
losses and maximum gradients. Measurements use CPython tracemalloc, including
traced NumPy allocations; they are not process RSS, browser heap or GPU memory.

| Default backward, bytes | 0.8 float32 | 0.9 float32 | 0.8 float64 | 0.9 float64 |
| --- | ---: | ---: | ---: | ---: |
| Retained after forward | 1,144,820 | 562,434 | 2,224,128 | 1,085,848 |
| Retained after backward | 1,179,468 | 55,338 | 2,291,264 | 86,480 |
| Peak through backward | 1,749,072 | 669,458 | 3,417,936 | 1,290,368 |

For this saved-activation chain, retained allocations after default backward fall
about 95–96%, with peak allocations about 62% lower. This is a fixture-specific
result. With explicit retention, 0.9 holds 595,164 / 1,152,224 bytes after the first
float32/float64 backward, falling to 53,236 / 86,008 bytes after a final unretained
pass. Graph history, input storage and live output storage still exist.
Use [bench_graph_memory.py](../../packages/paperleet-torch/benchmarks/bench_graph_memory.py)
to reproduce the allocation fixture.

Timings compare frozen main `fe09cfa6ae2a792f1e93e6b107164799967ca1be` / 0.8.0
with 0.9.0. All 31 candidate sources and benchmark scripts are hashed; the
baseline has 30 sources. CPU uses Python 3.12.11 / NumPy 2.5.4 with
OMP/MKL/OpenBLAS environment variables set to 1. Node v25.9.0 and Chrome 155.0.8059.40 use Pyodide
0.29.4 / NumPy 2.2.5. Fresh processes/Workers run sequentially without overlapping
local correctness/build jobs. CPU uses baseline→candidate→candidate→baseline.
Each case has three warmups and eleven batches of 120 calls. Balanced Node
controls use 360 calls per batch. All individual samples are retained.

CPU configuration clarification: these variables record environment settings;
they do not establish Apple Accelerate's threading policy. Historical samples
remain unchanged. The [0.10 record](torch-runtime-iteration-10.md) describes the
later BLAS provider audit and directly verified Accelerate controls.

The 16→32→8 MLP fixture includes gradient reset, forward, backward and SGD update
on a 16-row batch. Core fixtures exclude imports and construction. AdamW uses a
fixed gradient, so its timing is an update control rather than graph execution.
No-op conversion is below Chrome timer resolution with this batching and is not
used for speedup claims. Measurements assess these workloads, not cold starts,
end-to-end user submissions, training convergence or GPU performance.

| Common workload, float32 (µs) | CPU 0.8 → 0.9 | Node WASM 0.8 → 0.9 | Chrome WASM 0.8 → 0.9 |
| --- | ---: | ---: | ---: |
| Attention forward/backward | 223.3 → 227.1 | 2,661.4 → 2,669.2 | 2,689.2 → 2,685.8 |
| Linear forward/backward | 46.93 → 49.07 | 1,360.2 → 1,370.2 | 1,383.3 → 1,382.5 |
| MLP training step | 67.73 → 71.32 | 122.1 → 132.5 | 127.5 → 135.8 |
| Basic slice backward | 43.70 → 44.11 | 114.3 → 108.8 | 116.7 → 109.2 |
| Repeated-index backward | 101.9 → 107.3 | 215.2 → 218.5 | 221.7 → 225.0 |
| AdamW fixed-gradient update | 77.94 → 77.62 | 303.2 → 303.7 | 300.8 → 295.8 |

The memory/routing checks have a measurable cost for small graph workloads. The
float32 MLP is about 5.3% slower on CPU and 6.5% on Chrome. Node's primary pair
measures about 8.5%; balanced fresh-process repeats measure about 5.3% (baseline
127.0–128.8 µs, candidate 134.0–135.3 µs). Float64 balanced Node MLP repeats are
about 5.4% slower. This is a documented compatibility/memory tradeoff, not an
across-the-board speed improvement. Chrome float32 slicing is about 6.4% faster;
balanced Node slicing is about 6.6% faster. Larger attention/linear browser cases
remain close in these samples.

The first metadata prototype allocated separate leaf nodes and weak owners for
every intermediate. Its Node MLP controls cost about 13–19% over baseline.
Direct leaf references and lazy weak owners reduced that cost. Initial source
snapshots and every measurement remain in local `initial-profiling` and
`before-final-integer-sum-guard` archives. A final guard then made integer-dtype
sum reject active autograd; final sources were frozen again and all timings and
release gates were rerun. Final results do not reuse an earlier source hash.

Balanced fresh Chrome Worker controls use baseline→candidate→candidate→baseline
and 360 calls per batch. Float32 MLP medians are 132.2–133.1 µs versus
141.7–142.2 µs (about 7% slower); float64 is about 6.9% slower on average.
Float32 slicing is about 5.9% faster, while float64 slicing varies around parity.
Repeated-index controls average about 3.9% / 2.8% slower in float32/float64, with
visible process variation. AdamW averages differ by −0.1% / +0.7%.

The primary final Chrome float64 AdamW run recorded 395.0 µs versus 304.2 µs
(about 30% slower). Both candidate balanced repeats instead record 301.9–306.4 µs,
against baseline 301.4–302.5 µs; the earlier source variant was also near parity.
The large primary discrepancy was not reproduced, and its cause is undetermined.
The original samples remain in the record; no isolated measurement is discarded
or used to claim a kernel speedup.

## Validation scope

The complete maintained Python suite passes **2,255 tests**, one optional Docker
test skipped and six warnings originating in native Torch, 137.17 seconds.
The native lesson gate passes **69 modules, three repetitions, 8,829 executions
and all 295 mutants caught**, 106.72 seconds. Web checks and the production build
pass. All 31 final source hashes match the timing freeze.
The first integration rebases UI-only main `728b3c9` without changing package,
Python, test, lesson or catalog trees. A concurrent main update `8a5d952` then
clarifies DPO readmes and its starter docstring; the executable starter AST is
unchanged. The final integration preserves those updates and regenerates the
canonical catalogs. Library/application/test trees and all 31 library hashes
remain unchanged. The 120 catalog/config/browser tests pass again, as does a
three-repeat DPO gate (162 executions, all five mutants). Affected web checks and
production build also pass again. The Pages asset-retention checker passes all
15 cases; previously tracked published assets remain available to cached pages.

Real pinned WASM runs five complete lesson reference/mutant/learner suites plus
the existing numerical/fallback regressions and **82 lifecycle contracts** across
seven files. Five fresh Chrome Workers run **35 original tests, catch all 20
mutants and pass the same 82 additional contracts**, asserting version 0.9.0 and
the actual `/judge/paperleet_torch` import location. Exported runtime files match
the canonical library sources, including the new graph module.

The independent installed wheel outside the checkout passes **100 tests** with
NumPy 1.26.4 and no native Torch. It is **92,330 bytes**, SHA-256
`e706e80e51e6d49a58175de3782e8b5880ac14666cadafbd7b8b0e4d7d9bba48`.
Application/vendor and standalone packaging/launcher isolation remains covered
by the maintained suite, including unchanged torch-importing scripts.

New coverage includes 129 native-reference cases and 10 NumPy-only workflows.
Independent tests check analytical multi-root gradients, named/unused results,
retained intermediates, default/explicit graph reuse, actual saved-array lifetime,
requested input routing, error recovery, integer sum modes and absence of a leaf
self-cycle. All previous model, optimizer, scheduler, normalization, attention,
forward AD and reverse-over-forward tests remain in the release gate.

Final timing samples, source hashes, retained-memory samples and gate results are
recorded in [iteration-9.json](../../packages/paperleet-torch/benchmarks/results/iteration-9.json).
Full local logs, earlier variants and fresh-Worker scripts remain under
`paperleet-wasm-evaluation-20261015/torch-iteration-9`. The 0.8 predecessor has
successful quality/Pages runs and live runtime bytes verified before this work.
The broader independent-library, core/efficiency and optional resident GPU goals
remain active. No license has been selected and no PyPI release is made.

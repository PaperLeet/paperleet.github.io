# Torch runtime iteration 62 — cheaper log determinant common paths

Version **0.62.0** reduces correction-array and predicate overhead in existing
logdet/slogdet implementations. It retains the 0.61 API, derivative, singular,
mutation and graph-lifetime contracts. Final integration base is
`8f878a5a295aa50e322926da8ba92b31297ae953`; the full CPU/frontend/catalog baseline is
`43241a31db9649a1616f564a5b10f8141d0fb8ee`. 9 affected checks at `8f878a5a295aa50e322926da8ba92b31297ae953` plus driver/policy/two-build gates

## Changes and preserved behavior

Real ordinary inputs no longer construct unused NaN-sign corrections. Runtimes
whose LAPACK misclassifies NaN pivots still scan the input, but construct
correction arrays only for batches that actually contain NaN. A scalar mask
already has a truth value, so single-matrix predicates avoid a second NumPy
reduction. Positive real logdet reuses the factorization's logabs value instead
of making an unnecessary `where` result. Negative masks are computed once and
used by backward/JVP only when needed. There is no approximate factorization,
clipping, regularization, cross-call tensor cache or relaxed numerical threshold.

The ordinary reverse-graph loop again uses one combined no-gradient/pruning
condition. Unused-edge release and grad-frontier handling remain unchanged;
None gradients are not converted into zero arrays. Existing saved-value guards,
shared slogdet-output lifetime, immutable JVP factor snapshots and column-major
raw VJP layouts remain in force.

Complex forward, real negative/singular/nonfinite values, batches and empty
layouts retain the same validated behavior. Native reference is frozen PyTorch
2.14.1, git `5c4886908584029761b579af026dcfb627c84070`. The scope and intentional
limits are documented in the
[0.61 report](torch-runtime-iteration-61.md): real first-order VJP/JVP, complex
forward only, pinned saved-factor reverse-over-JVP semantics, separate legacy
det cofactor/nonfinite-input contracts, and equivalent positive singleton
leaf-gradient strides. Complex AD, output buffers, arbitrary higher derivatives
and live-autograd-graph standard pickle remain outside the supported scope.
The existing strict NumPy-error-policy det diagnostic from 0.61 is also unchanged.

## Verification

- Frozen final numerical source: **40,970 CPU passes, 1 skip**; source hashes
  captured before and checked after the process.
- Actual Node/Pyodide and Chrome Worker: **7,340 independent contracts each**,
  all 54 suites, Pyodide 0.29.4 / NumPy 2.2.5. Installed wheel on Python 3.10.19 /
  NumPy 1.26.4: **7,353 passes**, native Torch absent.
- Public wheel/sdist and vendored wheel match all **59 Python modules plus two
  license resources**. Coexistence, uninstalling either owner, aliases, fresh
  process serialization and native Torch isolation pass.
- Identical 600-case qualification on CPU, Node, Chrome and the installed minimum
  environment. Each checks 1,764 numeric arrays and 360 None-gradient fields:
  480 native log cases, 54 ordinary native det cases, 48 separate legacy det
  cases and 18 retained restriction cases. Legacy cases are not labeled native
  passes. Complex NaN/Inf components are checked separately. Ten stored-gradient
  arrays use the documented equivalent singleton-stride contract.
- All **78 exported modules and 342 mutants** pass the original browser budgets;
  embedded contracts include all 54 suites. All 1,152 native catalog input hashes
  remain unchanged. The 12,757 earlier immutable payloads are byte-identical;
  158 new assets are reachable and all 161 active export/index hashes are recorded.
- Full frontend checks, policy checks and both build profiles pass at the recorded
  baseline; any later disjoint UI changes have explicit staged qualification above.
- Seven release-policy files remain byte-identical to canonical
  `13d2782a494415ba72bc604a0ef7e4c4f875ba25`. No CI trigger, tag, dispatch, release,
  package upload or deployment is initiated.

Development prototypes and their source snapshots are retained. No new tests
are added merely to mirror these implementation branches: existing independent
scalar/native oracles, saved-storage/lifetime tests, layout tests and full
application checks verify the unchanged contract.

## Paired performance against actual 0.61 APIs

Twelve fresh children (four CPU, four Node, four Chrome), **100 matched workflows
per child**: real determinant/log forward and weighted backward, complex forward
and graph controls. Both sides invoke their actual logdet/slogdet APIs. The old
0.60 `det.log` composition is not the baseline here. Common calibrated batches
last at least 20 ms, with five alternating samples. Tables show the median of
four child ratios and their range; below 1 is faster. Inputs/conversions/seeds
are outside timing; dispatch, graph construction, kernels, weighted backward,
gradient clearing and result destruction are inside. Heavy correctness checks
finish before timing. CPU thread caps are two. No native-Torch, GPU, cold-start
or host-wide-isolation claim is made.

Real logdet forward:

| Runtime | Operation | Dtype | Shape | Mode | Candidate / 0.61 | Four-child range |
|---|---|---|---|---|---:|---:|
| cpu | logdet | float32 | [1, 1] | forward | 0.720 | 0.706–0.745 |
| cpu | logdet | float32 | [2, 2] | forward | 0.707 | 0.640–0.733 |
| cpu | logdet | float32 | [64, 64] | forward | 0.864 | 0.816–0.875 |
| cpu | logdet | float64 | [1, 1] | forward | 0.692 | 0.673–0.762 |
| cpu | logdet | float64 | [2, 2] | forward | 0.680 | 0.660–0.691 |
| cpu | logdet | float64 | [64, 64] | forward | 0.854 | 0.756–0.877 |
| node | logdet | float32 | [1, 1] | forward | 0.738 | 0.687–0.746 |
| node | logdet | float32 | [2, 2] | forward | 0.778 | 0.741–0.845 |
| node | logdet | float32 | [64, 64] | forward | 0.899 | 0.822–0.920 |
| node | logdet | float64 | [1, 1] | forward | 0.688 | 0.666–0.724 |
| node | logdet | float64 | [2, 2] | forward | 0.704 | 0.669–0.739 |
| node | logdet | float64 | [64, 64] | forward | 0.894 | 0.885–0.929 |
| chrome | logdet | float32 | [1, 1] | forward | 0.735 | 0.719–0.738 |
| chrome | logdet | float32 | [2, 2] | forward | 0.711 | 0.673–0.729 |
| chrome | logdet | float32 | [64, 64] | forward | 0.905 | 0.901–0.906 |
| chrome | logdet | float64 | [1, 1] | forward | 0.719 | 0.696–0.745 |
| chrome | logdet | float64 | [2, 2] | forward | 0.710 | 0.708–0.719 |
| chrome | logdet | float64 | [64, 64] | forward | 0.898 | 0.872–0.935 |

64×64 logdet forward plus weighted backward:

| Runtime | Operation | Dtype | Shape | Mode | Candidate / 0.61 | Four-child range |
|---|---|---|---|---|---:|---:|
| cpu | logdet | float32 | [64, 64] | full_step | 0.940 | 0.940–0.950 |
| cpu | logdet | float64 | [64, 64] | full_step | 0.923 | 0.881–0.998 |
| node | logdet | float32 | [64, 64] | full_step | 0.968 | 0.899–0.974 |
| node | logdet | float64 | [64, 64] | full_step | 1.001 | 0.980–1.070 |
| chrome | logdet | float32 | [64, 64] | full_step | 0.959 | 0.950–0.991 |
| chrome | logdet | float64 | [64, 64] | full_step | 0.953 | 0.916–0.983 |

Unchanged graph controls:

| Runtime | Operation | Dtype | Shape | Mode | Candidate / 0.61 | Four-child range |
|---|---|---|---|---|---:|---:|
| cpu | graph | float32 | [1] | full_step | 1.005 | 0.982–1.013 |
| cpu | graph | float32 | [32] | full_step | 0.991 | 0.980–1.007 |
| cpu | graph | float64 | [1] | full_step | 0.996 | 0.761–1.051 |
| cpu | graph | float64 | [32] | full_step | 1.004 | 0.994–1.040 |
| node | graph | float32 | [1] | full_step | 0.998 | 0.975–1.118 |
| node | graph | float32 | [32] | full_step | 1.009 | 0.868–1.059 |
| node | graph | float64 | [1] | full_step | 0.974 | 0.936–1.004 |
| node | graph | float64 | [32] | full_step | 0.986 | 0.974–1.072 |
| chrome | graph | float32 | [1] | full_step | 1.014 | 1.008–1.026 |
| chrome | graph | float32 | [32] | full_step | 1.002 | 0.985–1.021 |
| chrome | graph | float64 | [1] | full_step | 1.005 | 0.942–1.026 |
| chrome | graph | float64 | [32] | full_step | 1.012 | 1.000–1.038 |

Largest matched ratios, including losses:

| Runtime | Operation | Dtype | Shape | Mode | Candidate / 0.61 | Four-child range |
|---|---|---|---|---|---:|---:|
| node | det | float32 | [64, 64] | full_step | 1.073 | 0.989–1.152 |
| chrome | det | float64 | [4, 32, 32] | full_step | 1.035 | 0.979–1.072 |
| node | det | float32 | [1, 1] | forward | 1.034 | 0.999–1.210 |
| chrome | det | float32 | [4, 32, 32] | full_step | 1.034 | 1.027–1.092 |
| node | det | float32 | [8, 8] | full_step | 1.032 | 0.958–1.075 |
| cpu | det | float64 | [4, 8, 8] | forward | 1.029 | 0.988–1.064 |
| chrome | det | float64 | [8, 8] | full_step | 1.025 | 1.018–1.040 |
| chrome | det | float32 | [1, 1] | forward | 1.024 | 0.970–1.065 |

All 300 runtime/workflow aggregate rows, 1,200 child workflow results, calibration
values and samples are retained. Float32 old/new gradients retain the separately
labeled 1e-5 normwise benchmark check; independent native qualification retains
its existing elementwise/special-value/schema checks without relaxed limits.

The original full-matrix Node float32 64×64 det full-step median is 1.073,
with a 0.989–1.152 child range. Four additional fresh Node diagnostic children
isolate that unchanged numerical path: median 0.993, range 0.986–1.006.
The determinant module, layout/Tensor/autograd support modules and solve-helper
ASTs are verified unchanged; the generic graph condition ordering is the
remaining relevant implementation difference. This isolated diagnostic does not
replace or relabel the original full-matrix result and does not prove universal
absence of regression. Both datasets, source-identity checks and the first
metadata-only diagnostic-harness failure are retained. Graph medians/ranges above
remain visible; no blanket graph speedup is claimed.

## Allocation and footprint

Warm CPU tracemalloc incremental tracked bytes for 64×64 logdet below. The
archived `warm-memory.py` primes both actual API paths before tracing. Inputs
and seeds predate the trace. First-cold diagnostics are separate. Forward
retained/peak and backward peak are distinct measurements, not process RSS,
LAPACK workspace, total tape bytes or WASM heap size.

| Variant | Operation | Dtype | Newly retained forward | Forward peak | Backward peak |
|---|---|---|---:|---:|---:|
| baseline | logdet | float32 | 2,774 | 68,192 | 187,344 |
| baseline | logdet | float64 | 2,246 | 34,928 | 138,192 |
| candidate | logdet | float32 | 2,722 | 68,232 | 187,244 |
| candidate | logdet | float64 | 2,190 | 34,968 | 138,088 |

The complete 24-row allocation table is in the machine report. Source/license
bytes change from 731,240 to 731,699;
local concatenated gzip from 169,434 to
169,571 bytes. Gzip is a footprint diagnostic, not
wheel or transfer size. There are no additional runtime dependencies.

## Backend assessment

Source/documentation review confirms that the current Tensor owns a synchronous
NumPy ndarray. Pyodide can expose a JS view of that WASM memory, but GPU mapping
is asynchronous and portable WGSL has no concrete f64 type. An initial opt-in
f32 GEMM/linear experiment therefore needs storage/lifetime and synchronization
boundaries, explicit fallbacks and end-to-end transfer/readback measurements.
This is an architectural inference, not a measured GPU speedup; CPU/WASM remains
the production path. Sources: [pinned Pyodide buffers](https://pyodide.org/en/0.29.4/usage/type-conversions.html#buffers),
[WGSL scalar types](https://www.w3.org/TR/WGSL/#scalar-types),
[WebGPU buffer mapping](https://www.w3.org/TR/webgpu/#dom-gpubuffer-mapasync).
The full review is archived as `webgpu-backend-notes.md`.

## Reproduction

[Machine report](../../packages/stridetorch/benchmarks/results/iteration-62.json)
and [raw harness](../../packages/stridetorch/benchmarks/results/iteration-62-harness.tar.gz).
Archive: 5,356,757 bytes; SHA-256 `d388b0b739594df3f4da5c99cab6321a0443dc12dad6b4293f3d863e1d215c1b`. Frozen baseline/candidate
sources, programs, native inputs/results, runtime plans, package artifacts,
receipts, prototypes and samples are included. The repository benchmark is
`packages/stridetorch/benchmarks/bench_determinants.py`.

The broader independent-library/core-coverage/backend objective remains active.

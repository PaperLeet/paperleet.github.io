# Torch runtime iteration 66 — selective shared inverse JVP reverse kernels

Version **0.66.0** reduces repeated inverse-JVP reverse work on Pyodide/WASM,
adds traversal-local pruning for validated joint reverse rules, and corrects
empty-direction mutation guards. Existing public inverse APIs are unchanged.
Final integration baseline is `4db2b71b8948eb4fe8156fc80a97461431c412d8`; full CPU/frontend/catalog
validation baseline is `71a5d39622417e4148f33583c81d36fc91360c14`. 12 affected checks plus driver/policy/two builds at `323c0c4b1f0daaecd2360aeae7a18c0f791580fb`; 15 affected checks plus driver/policy/two builds at `d15da4aae8ac636d12160d0edfd452e46dfd6515`; complete frontend suite (145 commands) plus driver/policy/two builds at `c7ce04865a6dd6e94b3942d3f740abf4dbd6b9e0`; 4 affected checks plus driver/policy/two builds at `0f972c0bb493117932c0806e62b01224e74ef47a`; 10 affected checks plus driver/policy/two builds at `c79995feeb0a92db933e8c50e3dd3b81d5c6f672`; 9 affected checks plus driver/policy/two builds at `4db2b71b8948eb4fe8156fc80a97461431c412d8`

## Shared work and measured dispatch

The previous inverse JVP computes the same upstream-gradient matrix product
separately in the inverse-result and direction reverse branches. A joint rule
now shares it when both inputs require gradients, the runtime is Emscripten,
and the matrix core is at least 32×32. This removes one matrix multiplication
from the complete reverse step. CPU, smaller cores and one-input cases retain
the independent rules. The returned primal alias, native multiplication
associations, column-major output axes and saved-value guards are preserved.

A development scan covers ten sizes from 1 through 64 and both real dtypes on
CPU, Node and Chrome. Unconditional joint routing saves work in large WASM
cores but adds Python dispatch cost in small cases. The selected cutoff is a
conservative measured dispatch choice for the pinned runtime, not a universal
hardware optimum. Prototype A, its slower CPU pilot/profile, single-pass
prototype B, the size scan and the final dispatch pilots remain archived with
their own source hashes. They are distinct from final-source formal timing.

The graph engine's joint-gradient validation and routing now use one pass.
Existing rules still receive construction-time active-input flags. Only rules
that explicitly opt in receive masks for the current gradient targets. The
shared inverse rule skips an unrequested matrix or direction branch; repeated
retained-graph calls recompute from each new seed. No upstream-gradient cache
or cross-traversal kernel result is retained. Storage-version guards are checked
before callbacks and remain in force even when a derivative branch is pruned.

## Empty directions and the allocation-scope correction

The new native corpus finds **six** prior mismatches: both dtypes, three target
selections, and an interleaved empty batch. The previous stride comparison treats
the empty direction as an independent copy and permits its mutation. Native
PyTorch canonicalizes that empty tangent without making independent saved
storage, so its version guard must remain. Nonempty directions still use the
native-compatible stride-copy rule. The fix passes the new corpus and all prior
1,080 mutation/layout cases; no existing guards are relaxed.

The original 0.65 CPU allocation fixture uses
`broadcast_to(...).astype(...)`. A later audit verifies that its 4×32×32
direction has element strides `(1,128,4)`, while the primal has `(1024,32,1)`.
The native dual uses a copy: mutating the original direction does not change
the dual tangent's version. The 2D 64×64 fixture has matching `(64,1)` strides
and shares the original direction's version. This is confirmed for float32/64
against native **PyTorch 2.14.1**, git
`5c4886908584029761b579af026dcfb627c84070`.

The [0.65 report](torch-runtime-iteration-65.md) is corrected accordingly. Its
archived numbers, source and raw archive are not rewritten. The prior claim
that all directions in that allocation diagnostic were contiguous and required
no copy was inaccurate. Its batch retained-allocation cost includes the required
direction snapshot. The original latency benchmark constructs directions
differently and is unaffected by this scope correction.

## Verification

- **41,682 CPU tests pass, 1 skip** with exact source hashes checked before and
  after execution. **86 new SDK tests** cover opt-in/default joint masks, frozen
  and shared inputs, nonleaf target execution, repeated seeds, guard-before-
  callback behavior, malformed derivative checks, independent diagonal inverse
  derivatives, both dispatch paths and empty interleaved guards.
- **8,052 contracts / 60 suites** pass separately on Node/Pyodide and Chrome
  Worker. The isolated Python 3.10.19 / NumPy 1.26.4 wheel passes **8,065 tests**,
  with native Torch absent. Public wheel/sdist, vendored wheel, coinstallation,
  native-Torch alias restoration and fresh-process serialization are checked.
- **480 new frozen contexts**, generated from ten groups / 30 exact input
  arrays, cover partial gradient targets, retained primal/tangent gradients,
  hooks, graph reuse/release and storage mutations. Two dtypes, C/F/interleaved
  directions, 2×2 and 32×32 matrices, batches and empties compare **784 numerical
  arrays and 240 None fields**. CPU default routing, CPU forced-shared routing,
  Node, Chrome and the minimum wheel all match native states and meaningful
  strides. All IEEE masks are exact. The 320 large-matrix arrays use explicit
  per-matrix relative max-norm limits `1e-5` (float32) / `3e-12` (float64);
  small normal-input arrays use componentwise `rtol=3e-5, atol=3e-7` or
  `rtol=3e-12, atol=3e-14`. The prior stricter extreme-input corpus is retained.
- Previous **600 determinant/log**, **398 singleton**, **696 determinant JVP**
  and **864 inverse** contexts remain passing on all four environments, along
  with **1,080 inverse mutation/layout**, **105 matmul guard** and **120 API**
  checks. Native, legacy singular-cofactor and unsupported-input expectations
  remain separate. The package contains 59 Python resources and LICENSE/NOTICE.
- The original **78 exported modules / 342 mutants** pass original browser budgets,
  including the full 60-suite independent contract run. **1,152 native catalog
  input hashes and 13,389 previous immutable payloads** remain unchanged. There
  are 158 new reachable payloads and 161 active export/index hashes.
- Full frontend checks and both build profiles pass on the original Node 25.9.0 run; the latest affected checks and both builds
  also pass under CI-matching Node 22.23.3 after the upstream fix recorded below. An initial frontend run
  exposed a stale test after upstream `71a5d396` made the navigation label
  “Workspace” and moved progress hints to its title. A local correction checks the
  stable bilingual label and localized title while retaining empty/completed/
  loading/disabled-state checks. Upstream independently landed its own correction
  in `323c0c4b`; the final checkout uses that upstream test without a local test
  diff, and reruns it in the later UI qualification.
  The original failure, targeted reproduction, corrected pass and full rerun
  are preserved. Any later disjoint UI qualification is recorded above.
- All 18 public/fixture runtime assets, five Node core files and 13 lock-pinned
  package wheels are revalidated. Pyodide is 0.29.4 / NumPy 2.2.5; development
  CPU uses Python 3.12.11 / NumPy 2.5.4. Host metadata and per-child load averages
  are retained; machine-wide isolation is not claimed.
- Seven release-policy files remain byte-identical to canonical
  `4db2b71b8948eb4fe8156fc80a97461431c412d8`. No CI trigger, tag, workflow dispatch,
  release, package upload or deployment is initiated.

## Final-source performance against actual 0.65

Twelve fresh children, four each CPU/Node/Chrome, alternate runtime order and
use five alternating common calibrated batches of at least 20 ms. Owned tests
and builds finish before timing; supported CPU thread caps are two. Every
child measures **170 actual prior-API workflows**: 100 determinant/log/graph,
44 inverse/JVP/matmul/complex forwards and 26 existing GroupNorm/classification
controls. All comparator APIs exist in 0.65. No new-feature composition or
candidate-only ratio is used. Fixtures, conversions and seeds precede timing;
dispatch, graph construction, kernels, weighted backward, clearing and result
destruction are included when applicable.

Eligible large WASM inverse-JVP complete steps:

| Runtime | Operation | Dtype | Shape / case | Mode | 0.66 / actual 0.65 | Four-child range |
|---|---|---|---|---|---:|---:|
| node | inverse_jvp | float32 | [64, 64] | full_step | 0.914 | 0.911–0.925 |
| node | inverse_jvp | float32 | [4, 32, 32] | full_step | 0.929 | 0.919–0.933 |
| node | inverse_jvp | float64 | [64, 64] | full_step | 0.912 | 0.911–0.915 |
| node | inverse_jvp | float64 | [4, 32, 32] | full_step | 0.930 | 0.927–0.961 |
| chrome | inverse_jvp | float32 | [64, 64] | full_step | 0.915 | 0.906–0.918 |
| chrome | inverse_jvp | float32 | [4, 32, 32] | full_step | 0.931 | 0.928–0.934 |
| chrome | inverse_jvp | float64 | [64, 64] | full_step | 0.912 | 0.910–0.918 |
| chrome | inverse_jvp | float64 | [4, 32, 32] | full_step | 0.935 | 0.932–0.938 |

Small WASM complete steps, retaining independent routing:

| Runtime | Operation | Dtype | Shape / case | Mode | 0.66 / actual 0.65 | Four-child range |
|---|---|---|---|---|---:|---:|
| node | inverse_jvp | float32 | [1, 1] | full_step | 1.001 | 0.995–1.008 |
| node | inverse_jvp | float32 | [2, 2] | full_step | 1.010 | 1.006–1.016 |
| node | inverse_jvp | float64 | [1, 1] | full_step | 1.007 | 1.000–1.011 |
| node | inverse_jvp | float64 | [2, 2] | full_step | 1.000 | 0.995–1.009 |
| chrome | inverse_jvp | float32 | [1, 1] | full_step | 1.008 | 1.004–1.015 |
| chrome | inverse_jvp | float32 | [2, 2] | full_step | 1.014 | 1.012–1.024 |
| chrome | inverse_jvp | float64 | [1, 1] | full_step | 1.011 | 1.000–1.023 |
| chrome | inverse_jvp | float64 | [2, 2] | full_step | 1.010 | 1.008–1.020 |

CPU inverse-JVP complete steps, retaining independent routing:

| Runtime | Operation | Dtype | Shape / case | Mode | 0.66 / actual 0.65 | Four-child range |
|---|---|---|---|---|---:|---:|
| cpu | inverse_jvp | float32 | [1, 1] | full_step | 1.004 | 0.998–1.048 |
| cpu | inverse_jvp | float32 | [2, 2] | full_step | 1.010 | 1.004–1.015 |
| cpu | inverse_jvp | float32 | [64, 64] | full_step | 1.008 | 0.995–1.016 |
| cpu | inverse_jvp | float32 | [4, 32, 32] | full_step | 0.998 | 0.987–1.025 |
| cpu | inverse_jvp | float64 | [1, 1] | full_step | 1.002 | 0.990–1.015 |
| cpu | inverse_jvp | float64 | [2, 2] | full_step | 1.012 | 0.998–1.019 |
| cpu | inverse_jvp | float64 | [64, 64] | full_step | 0.994 | 0.973–1.015 |
| cpu | inverse_jvp | float64 | [4, 32, 32] | full_step | 1.000 | 0.991–1.006 |

Largest median ratios among other inverse/matmul/complex controls:

| Runtime | Operation | Dtype | Shape / case | Mode | 0.66 / actual 0.65 | Four-child range |
|---|---|---|---|---|---:|---:|
| cpu | inverse | float64 | [1, 1] | full_step | 1.018 | 0.990–1.022 |
| chrome | inverse | float32 | [2, 2] | forward | 1.015 | 0.996–1.023 |
| chrome | inverse | float64 | [2, 2] | full_step | 1.011 | 1.003–1.013 |
| chrome | inverse | float32 | [2, 2] | full_step | 1.011 | 0.997–1.016 |
| cpu | matmul | float64 | [64, 64] | full_step | 1.010 | 1.001–1.024 |
| cpu | inverse | float64 | [1, 1] | forward | 1.008 | 1.004–1.011 |
| cpu | inverse | float32 | [4, 32, 32] | full_step | 1.008 | 1.003–1.016 |
| cpu | inverse | float64 | [64, 64] | full_step | 1.008 | 0.971–1.017 |

Largest median ratios among the 100 prior determinant/log/graph workflows:

| Runtime | Operation | Dtype | Shape / case | Mode | 0.66 / actual 0.65 | Four-child range |
|---|---|---|---|---|---:|---:|
| cpu | det | float64 | [4, 8, 8] | full_step | 1.026 | 1.009–1.056 |
| cpu | det | float64 | [8, 8] | full_step | 1.024 | 0.997–1.047 |
| cpu | det | float64 | [64, 64] | full_step | 1.017 | 1.010–1.022 |
| cpu | logdet | float64 | [1, 1] | full_step | 1.016 | 1.012–1.020 |
| cpu | det | float32 | [4, 8, 8] | full_step | 1.016 | 1.000–1.022 |
| chrome | logdet | float32 | [1, 1] | forward | 1.015 | 0.997–1.034 |
| chrome | det | float32 | [2, 2] | forward | 1.015 | 0.992–1.023 |
| chrome | slogdet | float64 | [1, 1] | full_step | 1.014 | 0.992–1.035 |

Largest median ratios among 26 GroupNorm/classification controls, including
all-input, affine-only, bias-only, frozen-label and weighted-soft-target cases:

| Runtime | Operation | Dtype | Shape / case | Mode | 0.66 / actual 0.65 | Four-child range |
|---|---|---|---|---|---:|---:|
| node | classification | float64 | index_8 | index | 1.009 | 1.002–1.013 |
| chrome | classification | float64 | index_64 | index | 1.007 | 1.000–1.017 |
| node | groupnorm | float64 | forward | forward | 1.006 | 0.999–1.007 |
| cpu | groupnorm | float64 | forward | forward | 1.004 | 1.000–1.007 |
| chrome | groupnorm | float32 | forward | forward | 1.004 | 0.982–1.066 |
| chrome | classification | float64 | index_8 | index | 1.004 | 0.993–1.011 |
| node | groupnorm | float32 | mlp_train | train | 1.003 | 0.980–1.009 |
| chrome | groupnorm | float64 | forward | forward | 1.003 | 0.997–1.012 |
| chrome | classification | float32 | index_8 | index | 1.002 | 0.985–1.008 |
| cpu | classification | float64 | index_8 | index | 1.001 | 0.990–1.025 |

The official GroupNorm fixtures retain independent Tensor-composition checks;
classification fixtures retain their independent float64 formula checks.
Only actual library APIs are timed. A preliminary single-child classification
outlier of 31% did not reproduce in the four-child development recheck; both
the outlier and recheck remain archived. Final tables above use the twelve
fresh final-source children, retaining all samples and losses. There is no
native-Torch/GPU, cold-start or universal speedup claim.

## Allocation diagnostics and reproduction

Thirty-two warm CPU rows explicitly separate C/F 2D directions and C/interleaved
batch directions against actual 0.65 APIs. Actual primal/direction element
strides and the direction-copy predicate are recorded per row. Another 32 rows
force the shared WASM route on CPU to inspect its allocation behavior. This is
an explicitly marked CPU proxy, not a WASM heap measurement. The 24 prior
determinant/log diagnostics retain warm and cold-first results separately.

The shared-route proxy's JVP rows show forward retained/peak bytes and separately
traced backward peaks. Inputs/directions/weights precede tracing; the forced
runtime override is recorded. Necessary snapshots are kept for F/interleaved
directions. These counts are not total tape bytes, RSS, LAPACK workspace or
WASM heap size, and no universal memory saving is claimed:

| Dtype | Shape | Direction layout / strides | Prior / new retained bytes | Prior / new forward peak | Prior / new backward peak |
|---|---|---|---:|---:|---:|
| float32 | [64, 64] | C / [64, 1] | 55,304 / 55,216 | 134,880 / 134,880 | 87,536 / 104,480 |
| float32 | [64, 64] | F / [1, 64] | 71,744 / 71,656 | 134,880 / 134,880 | 87,504 / 104,448 |
| float32 | [4, 32, 32] | C / [1024, 32, 1] | 55,512 / 55,312 | 85,584 / 85,584 | 87,448 / 104,520 |
| float32 | [4, 32, 32] | interleaved / [1, 128, 4] | 71,968 / 71,824 | 86,368 / 86,368 | 87,416 / 104,432 |
| float64 | [64, 64] | C / [64, 1] | 104,456 / 104,368 | 135,240 / 135,240 | 169,320 / 202,648 |
| float64 | [64, 64] | F / [1, 64] | 137,280 / 137,192 | 168,136 / 168,136 | 169,304 / 202,632 |
| float64 | [4, 32, 32] | C / [1024, 32, 1] | 104,664 / 104,464 | 135,376 / 135,376 | 169,280 / 202,736 |
| float64 | [4, 32, 32] | interleaved / [1, 128, 4] | 137,504 / 137,360 | 168,288 / 168,288 | 169,288 / 202,688 |

The shared-route CPU proxy's backward peak rises about 19–20%, while retained
bytes change by less than 0.4% and forward peaks are essentially unchanged.
Default CPU inverse-JVP allocation rows are unchanged. Sharing saves arithmetic
but this implementation has an observable temporary-lifetime cost; lowering that
peak remains further work. These proxy numbers do not establish a WASM heap
increase of the same percentage.

[Machine report](../../packages/stridetorch/benchmarks/results/iteration-66.json)
and [raw harness](../../packages/stridetorch/benchmarks/results/iteration-66-harness.tar.gz)
contain source/test snapshots, frozen native inputs/outputs, prototype snapshots,
profiles, failed development pilots, corrected scope audit, package artifacts,
qualification programs, receipts and all timing samples. Archive: 24,483,830
bytes; SHA-256 `b21c9b9dd185237aca0a4f7de3172b2d6d4460f6792a083aae4386170665788b`.

Supported numerical execution changes only in the CPU/WASM graph and inverse
rules. The separate capability probe does not enable a WebGPU backend or claim
a GPU performance gain.
The wider independent-library efficiency/core-coverage/backend objective remains active.

## Latest 82-module catalog qualification

Upstream `c79995fe` adds top-p draws, multi-turn SFT masks, continual-learning
metrics and EWC. The prior 1,152 native exercise inputs and exact 61 runtime
resources remain unchanged. The four new modules pass 471 native cases across
three reference/mutant repeats and 30 WASM reference tests; all 17 new mutants
are detected within the original browser budgets. Combined coverage is 82
modules / 359 mutants, from the full original 78-module run plus this four-module
extension, rather than a fresh full-82 rerun.

The latest export preserves all 13,555 pre-existing payloads in the staged
snapshot, including the eight new upstream payloads, and adds eight reachable
runtime-specific payloads. The tree contains 13,563 immutable payloads and 169
active asset/index hashes. Ten affected frontend/bundle checks, the driver and
release policy checks, two local build profiles and 25 catalog tests pass.

The pre-existing upstream Pages run 39082894813 failed its Node 22.23.3 size
check before deployment. This is reproduced locally on the exact Pages-profile
artifacts: 409,388 raw bytes and 133,260 gzip bytes, versus limits of 409,600
and 133,120 bytes. Node 25.9.0 compresses those same artifacts to 132,726 bytes
and passes. The seven canonical release-policy files and size budgets remain
unchanged; no workflow retry or deployment was initiated. Local Node 25 build
success does not imply that the current Node 22 release gate passes.

The reusable [WebGPU bridge probe](../../packages/stridetorch/benchmarks/webgpu_bridge.mjs)
and [sandbox feasibility record](torch-webgpu-bridge-feasibility.md) also
record a successful scalar Python/GPU roundtrip in a secure Worker, unavailable
WebGPU in the tested opaque-origin Worker, and the asynchronous Python-entry
requirement. They do not enable a GPU backend or claim a performance gain.

## Upstream Node 22 build correction

Upstream `4db2b71b` separates shared profile models from UI-only modules. After
preserving all 187 owned paths byte-for-byte, nine affected UI/publishing checks,
the driver/policy checks and both build profiles pass with Node 22.23.3.
The default initial graph is 404,949 raw / 131,575 gzip bytes; the production
Pages graph is 406,806 raw / 132,118 gzip bytes. Both satisfy the original budgets.
The preceding c799 failure remains historical evidence, not a current limitation.

Upstream also adds the exact production Pages build to the quality job and
asserts parity of deployment variables. This iteration preserves those changes.
The CI trigger header is unchanged: tags/manual dispatch only; exact current-main
SHA gates and the Pages workflow remain intact. The seven policy hashes now pin
this reviewed upstream identity; the earlier identity is retained for old-stage
provenance. No push-triggered CI, manual retry or deployment is introduced here.

# Torch runtime iteration 51 — Attention validation and selective saved buffers

Version 0.51.0 reduces repeated attention metadata work and selectively retains
ordinary VJP operands. It preserves 0.50's GQA, dropout, forward/mixed AD and signed
balanced Q/K prescaling. The broader independent-library, core API/efficiency and
optional GPU-backend objective remains active.

## Implementation and memory scope

- Move repeated shape/dtype validation and default-scale calculation into one
  bounded 64-entry metadata cache. Keys contain shapes, canonical dtype strings
  and configuration; no NumPy dtype metadata, tensor, activation, seed, gradient
  or RNG state is cached. Tensor types and scalar options are checked every call.
- Replace generator-based hot-path checks with direct checks. A no-gradient call
  constructs its output without constructing an unused joint pullback. Saved
  versions, dtype rules, scalar validation and forward-tensor handling remain.
- Singleton K/V heads use normal matrix broadcasting in ordinary and forward AD,
  avoiding extra grouped reshapes and graph nodes. Other GQA heads retain grouped
  contraction and reduction without materialized repeated K/V heads.
- Save scaled K only when Q gradients need it, scaled Q only when K gradients
  need it, and V values only when logits gradients need them. Pullback needs stay
  captured at graph construction even when leaf flags later change.
- For a dropout VJP with Q/K/mask gradients, retain probabilities and one sampled
  multiplier; regenerate their product only for V gradients. A V-only pullback
  retains just the product. Retained backwards reuse the original multiplier and
  never draw RNG samples. Original input/mask version guards are preserved.

| Ordinary fused VJP | 0.50 unique score-shaped arrays | 0.51 |
|---|---:|---:|
| No dropout, all gradients | 1 | 1 |
| Dropout, all gradients | 3 | 2 |
| Dropout, V gradient only | 3 | 1 |

The 192 source-identified tape observations cover both dtypes, all 16 active-input
subsets and probabilities 0/.25/1. Weak-reference tests independently track
probabilities, the multiplier, the used probability product and scaled operands.
These are retained logical arrays in the fused first-order pullback, not complete
graph storage, process peak RSS or a compositional JVP/mixed-AD tape bound. Active
leaf storage can remain live for accumulation; the original-array lifetime probe
records that behavior explicitly. Transient products can be created in backward.
Score/probability sizes still grow with query length × key length; no streaming
attention, FlashAttention or GPU execution is claimed.

## Local validation

| Check | Result |
|---|---|
| Complete repository/library CPU suite | 39,813 passed, 1 skipped; 269.30s |
| New efficiency regression suite | 131 tests: 96 independent scalar dropout/VJP cases, 24 lifetime cases, 11 cache/type/value cases |
| Existing attention regression suite | 161 tests; 292 focused total |
| Native math primitive comparison | 204 cases, 2,592 arrays |
| Selected extreme finite-logit comparison | 32 cases, 208 arrays |
| Native mixed-AD encoder/SGD/checkpoint comparison | 200 arrays |
| Actual Node/Pyodide 0.29.4 / NumPy 2.2.5 | 6,184 contracts, 48 suites |
| Actual Chrome 155.0.8059.82 Worker | 6,184 contracts, identical sources/tests |
| Installed Python 3.10.19 / NumPy 1.26.4 standalone wheel | 6,197 passed; native Torch absent |
| Both distributions and independent uninstall | Exact 55 source/license resources, separate ownership |
| Original native PaperLeet catalog | 78 problems, 10,218 executions, 342 mutants, three repeats; 468.378s / 600s |
| Final exported WASM catalog | 78/78, 342 mutants, 6,184 contracts; 248.744s |
| Frontend and both build profiles | Five successful local receipts |
| Release policy | Seven files byte-identical to canonical `13d2782a494415ba72bc604a0ef7e4c4f875ba25` |

Independent scalar loops calculate grouped softmax, dropout values and every VJP
for each active-input subset, including fully-masked rows and repeated seeds.
Warm-cache tests change values, dtypes, feature/head shapes and scalar options.
The dtype-metadata test proves that a user object attached to a NumPy dtype is
collected after caller references disappear. Cache capacity stays bounded at 64.

Frozen native oracle arrays are reused from iteration 50's separately executed
PyTorch 2.14.1 math backend and checked against final 0.51 source. The primitive
set uses a native compositional safe-softmax reference for fully-masked mixed AD;
the 496 official-SDPA NaN-containing mixed arrays and their 100 cases remain
explicitly classified in the original oracle record. Every finite official entry
and all official values/JVPs/first VJPs are checked there. The stable zero mixed
derivative extension, strict [0,1] dropout probability rule and distribution-faithful
library RNG stream remain as documented in the [0.50 record](torch-runtime-iteration-50.md).
No native seeded dropout bit-stream equivalence is asserted.

The unchanged original native catalog receipt is reused after checking all 1,152
native catalog/runner/test/configuration file identities. Final source CPU,
Node/Chrome, package and exported-WASM checks run on the final adapter. Native
tolerances remain rtol 3e-5/atol 2e-5 for float32 primitive arrays, 5e-5/4e-6 for
encoder training; float64 uses 1e-11/8e-12 and 2e-11/2e-12. Selected extreme cases
use tighter checks. README wording was clarified to distinguish pullback arrays
from live leaf storage; the public package was rebuilt and minimum-wheel/coinstall
checks repeated. Mathematical source/test hashes did not change for that recheck.

## Performance versus 0.50

The primary portable benchmark loads independent actual 0.50 and 0.51 snapshots.
Both use their builtin GQA. It runs 36 primitive flows across ordinary/GQA/dropout
attention and forward/reverse/JVP, plus four full encoder SGD flows with/without
dropout in both dtypes. Values, all VJPs, JVPs and reverse-over-forward derivatives
are compared before timing; three seeded matching optimizer updates are checked.
Timed SGD does not copy parameters back for diagnostics on every update.

A same-source fully repeated K/V control covers forward/reverse/SGD. It does not
claim unsupported repeat_interleave JVP. Dropout qualification resets identical
seeds; timed calls use the same distribution without per-call seed-reset overhead.
All timing samples and regressions are retained. Positive percentages mean less time.

| Runtime | Workload | Mode | Median time reduction vs 0.50 | Range across flows/dtypes |
|---|---|---|---:|---:|
| NumPy CPU | ordinary | forward | 5.0% | 1.3% to 11.1% |
| NumPy CPU | ordinary | reverse | 2.6% | 0.7% to 4.0% |
| NumPy CPU | ordinary | jvp | 2.1% | 1.3% to 3.4% |
| NumPy CPU | GQA | forward | 1.6% | 0.4% to 4.1% |
| NumPy CPU | GQA | reverse | 0.6% | -1.0% to 6.0% |
| NumPy CPU | GQA | jvp | 0.8% | -4.4% to 4.4% |
| NumPy CPU | dropout GQA | forward | 1.6% | 1.6% to 1.6% |
| NumPy CPU | dropout GQA | reverse | -0.7% | -1.7% to 0.3% |
| NumPy CPU | dropout GQA | jvp | 3.0% | 2.8% to 3.1% |
| NumPy CPU | encoder SGD p=0.0 | full step | 1.1% | -0.5% to 2.7% |
| NumPy CPU | encoder SGD p=0.25 | full step | 6.0% | 1.7% to 10.4% |
| Node/Pyodide | ordinary | forward | 6.0% | -1.2% to 9.7% |
| Node/Pyodide | ordinary | reverse | 1.7% | -0.1% to 4.4% |
| Node/Pyodide | ordinary | jvp | -1.4% | -2.6% to 0.5% |
| Node/Pyodide | GQA | forward | 1.2% | -1.6% to 7.9% |
| Node/Pyodide | GQA | reverse | -0.1% | -0.9% to 2.1% |
| Node/Pyodide | GQA | jvp | 1.5% | -0.3% to 4.9% |
| Node/Pyodide | dropout GQA | forward | -0.6% | -1.0% to -0.1% |
| Node/Pyodide | dropout GQA | reverse | -0.2% | -2.1% to 1.7% |
| Node/Pyodide | dropout GQA | jvp | 1.0% | 0.9% to 1.1% |
| Node/Pyodide | encoder SGD p=0.0 | full step | 2.6% | 2.2% to 3.0% |
| Node/Pyodide | encoder SGD p=0.25 | full step | 4.4% | 4.1% to 4.8% |
| Chrome Worker | ordinary | forward | 6.7% | -1.6% to 14.3% |
| Chrome Worker | ordinary | reverse | 0.2% | 0.0% to 10.7% |
| Chrome Worker | ordinary | jvp | 1.1% | -0.3% to 3.6% |
| Chrome Worker | GQA | forward | 0.5% | -0.2% to 4.0% |
| Chrome Worker | GQA | reverse | 0.6% | -0.9% to 4.8% |
| Chrome Worker | GQA | jvp | 1.5% | -0.3% to 4.3% |
| Chrome Worker | dropout GQA | forward | 0.7% | 0.0% to 1.3% |
| Chrome Worker | dropout GQA | reverse | -1.2% | -2.1% to -0.3% |
| Chrome Worker | dropout GQA | jvp | 1.7% | 1.7% to 1.8% |
| Chrome Worker | encoder SGD p=0.0 | full step | 0.5% | 0.0% to 0.9% |
| Chrome Worker | encoder SGD p=0.25 | full step | 0.8% | 0.0% to 1.6% |

The primary complete-SGD time reductions, taking the median across both dtypes
and dropout settings, are 2.2% on CPU, 3.5% on Node/Pyodide and 0.5% on Chrome.
Some individual flows still regress; every range and underlying sample is retained.

## Regression comparison versus 0.49

A separate unchanged 22-flow harness compares actual 0.49 and 0.51 source.
It preserves singleton-head broadcasting and repeats only non-singleton heads that
must expand. This covers 0.49-supported forward/reverse and original encoder SGD
workloads; it does not simulate its missing dropout/JVP support. Its original
timed diagnostic copies remain unchanged, so it is separate from the primary
training benchmark. The comparison measures whether 0.50's observed regressions
are recovered under the same current runtime, without mixing old timing sessions.

| Runtime | Workload | Median time reduction vs 0.49 | Range across flows/dtypes |
|---|---|---:|---:|
| NumPy CPU | ordinary | -3.9% | -8.2% to 0.4% |
| NumPy CPU | GQA | 1.6% | -4.0% to 2.9% |
| NumPy CPU | encoder SGD | -1.1% | -1.5% to -0.6% |
| Node/Pyodide | ordinary | -2.5% | -5.6% to 1.5% |
| Node/Pyodide | GQA | 0.7% | -3.7% to 5.4% |
| Node/Pyodide | encoder SGD | -0.9% | -2.1% to 0.3% |
| Chrome Worker | ordinary | -0.6% | -14.3% to 0.0% |
| Chrome Worker | GQA | 1.0% | -5.1% to 6.0% |
| Chrome Worker | encoder SGD | -1.8% | -1.8% to -1.8% |

Under this unchanged legacy protocol, complete-SGD medians remain 0.9%–1.8%
slower than 0.49. This revision does not fully erase the older-version regression.
The primary and legacy training protocols are kept distinct.

There are 24 fresh child invocations: four per runtime per benchmark phase, with
alternating phase/runtime/method order, five samples and eight calls per sample.
CPU numerical-library threads are capped at two; Node and actual Chrome use the
pinned Pyodide runtime. Runtime metadata and load observations are retained. This
shared-host evidence does not establish exclusive capacity, universal speed,
native Torch/GPU speed, cold-start or process peak-memory improvement. Pilots and
concurrent preflights are preserved separately from the final sequential run.

## Source and integration identity

All 53 Python modules plus LICENSE/NOTICE remain. Fifty-three old resources are
unchanged; `__init__.py` and `_attention.py` change. Source/license bytes change
from 655,234 to 656,717;
local concatenated gzip changes from 151,531 to 151,889.
These are source estimates, not wheel/network bytes or measured memory. Dependency
versions and lockfile are unchanged; fourteen pinned Pyodide files were rechecked.

All 11,019 published immutable payloads are preserved. The 158 new reachable
payloads yield 11,177 total; 161 active export hashes match the final source.
Concurrent upstream changes, when present, are preserved as exact Git blobs with
fresh affected frontend/build checks and recorded in `upstream_syncs`.

The Chrome harness owns a localhost fixture server on an ephemeral port and closes
it with its browser. It needs no pre-existing fixed-port server. Numerical source,
tests, native reference arrays, scalar/lifetime observations, command receipts,
logs and all primary/legacy timing samples are retained in `iteration-51.json`,
`iteration-51-integration.json` and four matching raw harness archives under
`packages/stridetorch/benchmarks/results`.

Broader core coverage, Multihead/Transformer modules, additional temporary-buffer
work and optional persistent GPU execution remain active. No tag, workflow
dispatch, release, PyPI upload or deployment is initiated by this work.

## Main adoption

Validated source `dc8dfe349be76be0f6fafae40f8e002949d99a94` was adopted on remote main by ordinary non-force push. The remote SHA was verified exactly; zero Actions runs were observed for this source SHA. All 55 source/license resources, 161 active export identities, 10,861 immutable payloads and seven canonical release-policy files were rechecked. The separate proof commit is verified outside the repository after its push. No tag,dispatch,release,PyPI upload or online deployment was initiated by this work.

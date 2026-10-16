# Lightweight torch iteration 15: WASM linear forward contractions

Repository integration version: `paperleet-torch` **0.15.0**. Apache-2.0 and
ordinary `import torch` launcher/host aliases remain available. All local
integration gates below passed before main publication. Exact remote Quality,
Pages and live-source verification also succeeded for `7d94e9bc`; the final
adoption proof is appended below.

## Why this kernel

A real Pyodide 0.29.4 / NumPy 2.2.5 cProfile run executed the unchanged Flow
Matching T3 experiment: 2,000 training steps, batch 256, the existing TimeMLP,
Adam, 2,000 sampling points, Euler integration and the original distribution
assertions. It passed. Of the 24.209 profiled seconds, linear forward owned
9.714 s and the existing linear pullback contractions 7.969 s. SiLU forward
owned 1.291 s. These diagnostic times are **not** performance comparisons.

Sixteen fresh, balanced ABBA Node/WASM kernel processes compared matmul with
einsum, flattened einsum, packing the transposed weight and NumPy inner.
Direct einsum reduced the larger hidden/sampling forward kernel times by about
54–56%; packing did not help. Small cases regressed. A separate four-process
40-case float32/float64 boundary study showed that even equal operation counts
do not guarantee a win: short feature vectors could regress substantially.
Every process and batch sample is retained, including outliers.

The adopted path therefore uses direct `...i,oi->...o` contraction only on
WASM, with at least 64 input features, at least 32,768 multiply-accumulate
positions, and contiguous feature strides for both operands. CPU BLAS,
shorter features, small workloads and other feature strides retain matmul.
Existing WASM linear pullbacks are preserved. The forward-mode rule remains
the existing differentiable Tensor composition.

## Complete-workflow measurement

All correctness/build producers were terminal before the timing series. Each
sample used a fresh Node process, or a fresh entire Chrome process and Worker.
Runtime loading/preparation was excluded; the **complete original eight-test
Flow Matching reference suite** was included, with no profiling. No training
steps, tests, mutants, numerical thresholds or browser budgets were changed.

| Runtime | Original 0.14 samples, seconds | Candidate samples, seconds | Median time reduction |
| --- | --- | --- | --- |
| Node 25.9.0 / Pyodide 0.29.4 | 26.2390, 23.5823 | 18.1588, 18.2152 | 26.99% |
| Chrome 155.0.8059.82 Worker | 23.6667, 24.8421 | 18.8466, 19.4110 | 21.13% |

Both series used baseline/candidate/candidate/baseline order. The baseline
spread is preserved; these are local point estimates, with two processes per
variant per runtime, not general hardware or remote deadline guarantees.

All eight complete suites passed. The direct contraction changes floating-point
reduction order: it does not produce bitwise-identical training. Within the
Node pair, the maximum recorded loss-curve difference was `4.76837158203125e-7`
and sample coordinate difference `1.7881393432617188e-6`. Reference samples were
identical. All mode proportions and the `0.12300000339746475` outlier fraction
were exactly equal, with all eight modes covered. MMD² changed from
`0.012716227709823613` to `0.012716226474186465`; the unchanged threshold is
`0.019074374055781274`.

## Source and package identity

The original benchmark baseline is `de4490c9`, the fully verified 0.14 release.
Main advanced to `569ba1ed` with course/favorites UI work. Its 35 Python runtime
modules and two license resources were still byte-identical to the frozen
baseline before adopting this change. Those upstream changes were fast-forwarded
and preserved. The later `a3e2809c` runtime-download retry merge was also
fast-forwarded after local producers terminated. It changed no Python/library,
problem, test, exported-data or real-WASM harness inputs; frontend checks and
the production build then included its new download-recovery checker.

The measured prototype and release runtime have exactly the same modules and
resources except `__init__.py`'s package version `0.14.0` → `0.15.0`. This bridge
is asserted in the measurement record. Fresh wheel and source archives each
contain the same 35 Python modules and the Apache-2.0/NOTICE resources as source.
A fresh actual Python 3.10.19 / NumPy 1.26.4 environment installed the wheel,
without native PyTorch, and passed **226 tests** in 4.83 s.

## Validation status

- Focused scalar-sum/native checks: **182 passed**. New coverage includes large
  vectors/batched matrices, feature-strided/Fortran inputs, frozen inputs,
  retained graphs with distinct seeds, reverse-over-forward differentiation,
  and input/weight/output saved-version mutation guards.
- Complete Python infrastructure/package tests: **17,103 passed**, one optional
  Docker skip, seven warnings, 251.86 s.
- Native catalog: **77 modules**, **9,987 test executions**, three repeats,
  all mutation checks passed, 251.629 s.
- Complete pinned WASM reference/mutation catalog: **77/77 passed**, 597.034 s;
  this includes the 206 standalone in-VM contracts. The local runner observed
  no browser-budget excess. These correctness gates ran concurrently, so their
  elapsed times are not controlled performance comparisons.
- All 77 visualization datasets generated; frontend `npm run check` and
  production `npm run build` passed, including the incoming download retry fix.
- All **5,182 published hashed assets** remain byte-identical. All **156 new
  active assets** are reachable. Four superseded, untracked and unpublished
  pre-visualization module snapshots were backed up and removed; active bytes
  were identical before/after this cleanup.
- Final standalone/private-vendored wheel and source archives carry the same
  35 runtime modules and LICENSE/NOTICE; installed paths are disjoint. Refreshed
  standalone README metadata is checked against source. The final executable
  module hashes equal the isolated wheel that passed 226 tests.

The [complete local validation record](../../packages/stridetorch/benchmarks/results/iteration-15-validation.json)
preserves raw catalog results, package/resource hashes, upstream invariance,
asset preservation/pruning evidence and log attestations. Final deployment
expectations were frozen before publication; remote results remain a separate
gate and are not claimed here.

The first new-test draft incorrectly required `None` instead of an allowed
zero primal gradient. An unscaled random-network test also exceeded an
arbitrary float32 absolute bound on both the existing CPU and WASM paths.
The corrected test checks the zero mathematical derivative and uses 0.1-scale
network weights; strict scalar-sum and native comparisons pass. The original
failed log is retained. The first cProfile run passed the experiment but failed
report export after asking the alias for an absent `__version__`; its rerun
uses the actual package version. Neither failed tooling run is a timing sample.

The [measurement record](../../packages/stridetorch/benchmarks/results/iteration-15.json)
contains all per-batch/per-process timing samples, source hashes, numerical
results, package identity and raw artifact attestations. Its accompanying
`iteration-15-harness.tar.gz` contains the exact diagnostic/experimental runners
and frozen input plans. Full local reports live under
`/Users/david/Developer/paperleet-wasm-evaluation-20261015/torch-iteration-15`.

## Exact remote adoption proof

Commit `7d94e9bc0bc8a4ec810b014c519389694fd88cc3` passed
[Quality38833400470](https://github.com/PaperLeet/paperleet.github.io/actions/runs/38833400470)
with successful native/frontend, browser-WASM and pilot jobs. Linux reports
**17,103 passed**, one optional Docker skip and ten warnings in 494.53 s.
The native catalog passes 77 modules, 9,987 executions, three repetitions and
336 mutants in 403.027 s. The real pinned WASM catalog passes **77/77** in
980.910 s, including the 206 standalone in-VM contracts.

This runner observed no browser-budget excess. Flow Matching preparation was
3.811 s, calibration **228.354 s**, learner **45.724 s**, and total **278.296 s**;
the unchanged calibration/run/total budgets are 240/60/300 seconds. These are
remote device observations, not a controlled comparison against another CI run.
Calibration still has limited headroom; no universal deadline guarantee follows.

Exact [Pages38833400206](https://github.com/PaperLeet/paperleet.github.io/actions/runs/38833400206)
also succeeded. The public deployment's 77 starters, 77 module details and
77 judge manifests match the frozen source expectations; all 156 immutable
assets and both 0.15 runtime/resource profiles match byte-for-byte. LICENSE and
NOTICE are checked in both runtime profiles. The remote observer, downloads and
live verifier were all observed terminal before starting the next iteration.

See the [remote adoption record](../../packages/stridetorch/benchmarks/results/iteration-15-remote-adoption.json)
for exact run IDs/jobs, raw catalog results, Flow Matching budgets, live payload
hashes, log attestations and process receipts. Subsequent main UI changes through
`0a6f0ecf` preserve all 35 numerical modules and two license resources unchanged.

# WASM matrix packing candidate after 0.80

An isolated candidate packs both matrix operands into contiguous reduction rows
before the existing eligible 2D WASM `einsum`. This is a promising optimization;
it has **not** replaced the supported runtime or been adopted on main.

The [subsequent six-process and memory qualification](torch-runtime-matmul-packed-rows-qualification.md)
selects this exact candidate for integration and records the remaining package
and application gates. Its repeated measurements supersede the preliminary
timing estimates below.

The numerical source is 0.80 with exactly one changed file, `_matmul.py`.
The existing dtype, size and exceptional-range guards stay in place, as do the
normal native CPU route and batched matrix implementation. The candidate replaces
`einsum('ik,kj->ij', a, b)` with `einsum('ik,jk->ij', contiguous(a), contiguous(b.T))`.
Output normalization remains unchanged. The
[candidate patch](../../packages/stridetorch/benchmarks/fixtures/matmul-packed-rows-candidate.patch)
is retained separately from production sources.

## Evidence so far

- Four existing matrix suites pass against the exact candidate: **3025** tests
  on NumPy 2.5.4, and **3022 passed / 3 skipped** on NumPy 1.26.4. These include
  randomized values, VJP/JVP/HVP, C/F/strided/broadcast layouts, signed zeros,
  exceptional-value fallback and saved-mutation guards.
- The complete **89-suite / 12212-test** SDK gate passes in actual Node/Pyodide
  0.29.4 with NumPy 2.2.5. Only the same 13 native standalone-process tests
  omitted from prior WASM gates are excluded. Tests and tolerances are unchanged.
- An initial 14-flow scan and a later 18-flow scan measure complete public
  forward, both input VJPs and three owned output copies. Each uses independent
  A0/A1 baseline imports, shared read-only operands, all six execution orders,
  calibration and GC control. All A/A medians stay within the predefined ±5%.
  Both scans are exploratory **single-process Node** measurements.

In the later scan, the candidate/original paired time ratios are:

| Flows | Rows | Candidate / original |
| --- | ---: | ---: |
| 128/256 float32/float64, strided | 4 | 0.540–0.604 |
| 128/256 float32/float64, broadcast rows | 4 | 0.851–0.950 |
| 128/256 C/F plus 512 float32 C | 9 | 0.891–0.932 |
| 64 float32 C, below the optimized route | 1 | 1.003 |

These results suggest approximately **40–46% lower time for strided inputs**,
**5–15% for broadcast rows**, and **7–11% for the measured dense C/F flows**.
They do not establish a general speedup or qualify this candidate for adoption.

The attempted expanded scan in iteration 343 aborted when it tried importing
negative-stride arrays with `from_numpy`, which the existing API rejects. Its
four partial rows and exception are retained. A fresh 344 fixture keeps all four
negative-stride cases as untimed external-import rejection controls and checks
their three underlying private-kernel products. All four controls pass; the
18 valid public flows remain measured. No API restriction was relaxed to obtain
a passing result.

## Remaining qualification

Before selecting a runtime change, extend rectangular/dispatch-boundary flows,
repeat balanced measurements in fresh Node and Chrome processes, measure peak
temporary allocations and lifetimes, and qualify the final source with current
application and package integration. Guard against packing copies erasing gains
for other shapes or increasing memory unacceptably. Any later guard or algorithm
change requires its own frozen candidate and evidence.

The [machine report](../../packages/stridetorch/benchmarks/results/matmul-packed-rows-candidate.json)
contains both scans, raw samples and all 65 candidate resource hashes. The
[raw archive](../../packages/stridetorch/benchmarks/results/matmul-packed-rows-candidate-harness.tar.gz)
contains 577 verified files, 1,673,735 bytes, SHA-256
`bca1c5b901765ff5ca34af5fc9047a43f894d35063778553e83199c623601004`.
It retains the original workspace paths and external runtime dependencies;
adapt these for replay. This candidate is independent of the explicit GPU
prototype. No production code, package version, CI trigger or deployment changes.

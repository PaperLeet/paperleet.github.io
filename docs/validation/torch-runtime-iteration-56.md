# Torch runtime iteration 56 — sequence factories

Version 0.56.0 improves `arange` and `linspace` correctness and adds `logspace`
to the independent NumPy CPU / Pyodide WASM library and PaperLeet's vendored runtime.
The package remains named StrideTorch pending a separate naming decision.
The broader efficiency, core API coverage, backend and independent-library objective remains active.

## Behavior

- Integer `linspace` truncates endpoint conversions and interpolated values toward zero.
  For example `torch.linspace(-1, 1, 4, dtype=torch.int64)` now returns `[-1, 0, 0, 1]`;
  the previous NumPy integer-floor result was `[-1, -1, 0, 1]`.
- Floating `arange` uses an adequately wide index computation before final storage conversion.
  At offset `1e8`, float32 output no longer collapses merely because a unit increment rounds away.
  Bounds determine output size before integer start/step truncation. Exact int64 bounds retain large offsets.
- Native positional/keyword overloads, dtype inference, scalar conversion, empty/singleton behavior,
  gradient leaf ownership and invalid-step checks are validated. Ordinary `arange` Tensor controls
  requiring reverse gradients fail with TypeError; Tensor-endpoint `linspace`/`logspace` JVP remains
  explicitly unsupported, matching the installed native target. Constant factories have zero JVP.
- Real and complex `logspace` support real bases, including negative/zero-base IEEE behavior.
  Singleton exponent conversion is intentionally distinct from multi-element integer endpoint conversion.
- Float16 `linspace` uses float32 intermediates. Five finite wide spans overflow in installed native
  Torch 2.14.1 but pass an independent rational oracle here. The opmath choice follows the
  [upstream CPU range-factory design](https://github.com/pytorch/pytorch/blob/5729a7fce9e6aca33d3c0fc4e63825e5457314b2/aten/src/ATen/native/cpu/RangeFactoriesKernel.cpp).
  These five cases are not counted as native numerical passes.
- Two-ended interpolation operates in one owned work array. Four fixed unit intervals with at most
  256 elements have a 64-entry immutable byte-template cache, bounded by 128 KiB numeric bytes
  plus Python metadata. Every result receives independent writable storage; arbitrary endpoints and
  Tensor objects are not retained. Factories cap output at 16 million elements.

## Verification

| Gate | Result |
| --- | --- |
| Full repository CPU tests, including SDK | 40,608 passed, 1 skipped |
| Actual Node Pyodide 0.29.4 / NumPy 2.2.5 | 6,978 passed |
| Actual Chrome 155.0.8059.82 worker, same Pyodide | 6,978 passed |
| Installed public wheel, Python 3.10.19 / NumPy 1.26.4 | 6,991 passed |
| Exported PaperLeet catalog | 78/78 modules, 342 mutants; two initial budget flags checked again while idle |
| Independent contracts inside exported attention runtime | 6,978 tests / 52 suites |
| Public/private wheel coexistence and independent uninstallation | Passed; exact 59-resource identities |
| Fresh-process module pickle restoration and `import torch` alias sequence checks | Passed for both distributions |
| Frontend checks, driver/release policy checks, normal and Pages-profile local builds | All five passed |
| Immutable catalog assets from previous main | All 11,809 unchanged; 158 new reachable assets |
| Active catalog/runtime/index identities | 161 checked |
| Canonical release policies | Seven byte-identical to `13d2782a494415ba72bc604a0ef7e4c4f875ba25` |

The new sequence suite adds 111 tests. All 59 source/license resources (57 Python modules,
LICENSE and NOTICE) are hash checked across source, public/private wheels and actual WASM runs.
Existing native catalog qualification is retained after rechecking all 1,152 native input hashes.
The original full WASM batch overlapped CPU/wheel qualification and flagged ddpm_sampling and
flow_matching for calibration/total time budgets. Original timings and flags remain in the archive.
Exactly those two cases were rerun sequentially after every other heavy gate and benchmark was
terminal; both CLI correctness checks and all original time budgets passed, with identical
bundle/runtime/source identities. The final budget qualification references those separate
receipts and does not recast the initial full run as a clean-budget run.
Public/private/native dtype state and alias restoration remain independently validated.

## Native differential qualification

The same 4,215-case driver and frozen native Torch 2.14.1 outputs run against the final source on
CPU, Node WASM and Chrome WASM. Each runtime reports exactly:

| Classification | Cases |
| --- | ---: |
| Native numerical comparisons | 2,587 / 2,601 arrays |
| Error/unsupported behavior with checked categories | 1,514 |
| Wide float16 spans checked against an independent rational oracle | 5 |
| Undefined out-of-range floating-to-integer casts, retained but excluded from portable numeric claims | 109 |

Dtype and shape match exactly. Integer comparisons are exact. Floating rtol/atol are
float16 `3e-3/2e-3`, float32/complex64 `2e-6/2e-7`, float64/complex128 `2e-13/2e-14`.
Real/imaginary NaN and signed-infinity masks are checked separately. For finite complex components,
error is bounded by `atol + rtol * abs(reference complex value)`, using the finite components of a
partially non-finite reference. This allows platform libm phase-rounding near a zero component;
it does not allow NaN to hide an incorrect infinity or finite component.

Error categories: 663 RuntimeError matches, 822 native NotImplementedError to typed unsupported,
19 TypeError matches, two ValueError matches, and eight unrepresentable native complex-half export
TypeErrors to typed unsupported. Unsupported cases are not numeric passes.

Diagnostic attempts remain in the raw archive: the initial browser harness lacked its index HTML;
it was fixed before the successful run. A stricter absolute component tolerance exposed complex64
negative-base phase differences across CPU/WASM libm; the final complex norm-scaled bound above
is explicit and used identically on all runtimes. Intermediate qualifier/working-directory failures
are retained alongside the final successful artifacts. The first manual minimum-version wheel
invocation omitted the shared BLAS thread caps. A CPU sample showed ten OpenBLAS threads and
parallel wait frames; only that owned test child was interrupted, its logs/sample were preserved,
and the same checks were rerun with the standard two-thread configuration.

## Performance

Baseline is validated 0.55.0 (`32359d74e56cda83411cf53183d58dc65d88ee28`), candidate is 0.56.0.
Four fresh children per runtime, five samples per case, alternate baseline/candidate order.
Batch sizes are calibrated outside timing to at least 20 ms per variant, then shared by both variants
within each case. This avoids zero-duration samples from the browser's coarsened clock.
Each child has 23 matched valid `arange`/`linspace` cases and six candidate-only `logspace` cases.
Inputs with known incorrect baseline rounding are excluded from speedup comparisons.
Construction/destruction is timed; correctness, warmup and GC are outside timing.

Representative float32 results: candidate latency / candidate-to-baseline time ratio, followed by
full ratio range across four processes. Ratios below one mean faster. New logspace has no baseline ratio.

| Factory / elements | CPU | Node WASM | Chrome WASM |
| --- | --- | --- | --- |
| arange / 16 | 3.61 μs / 2.219× (2.200–2.243) | 8.78 μs / 2.295× (2.278–2.306) | 8.62 μs / 2.258× (2.234–2.278) |
| arange / 4,096 | 4.69 μs / 1.739× (1.725–1.754) | 10.09 μs / 2.006× (1.974–2.016) | 9.90 μs / 1.976× (1.957–1.980) |
| arange / 1,000,000 | 253.17 μs / 1.007× (1.007–1.009) | 279.41 μs / 1.018× (1.017–1.018) | 282.42 μs / 1.023× (1.020–1.025) |
| linspace / 16 | 3.05 μs / 0.856× (0.846–0.893) | 7.65 μs / 0.978× (0.975–0.986) | 7.69 μs / 0.945× (0.927–0.957) |
| linspace / 4,096 | 8.65 μs / 1.509× (1.500–1.515) | 20.03 μs / 1.550× (1.542–1.556) | 20.17 μs / 1.553× (1.539–1.559) |
| linspace / 1,000,000 | 400.98 μs / 0.688× (0.674–0.689) | 1024.99 μs / 0.914× (0.908–0.919) | 1015.62 μs / 0.919× (0.910–0.921) |
| logspace / 16 | 8.78 μs | 20.05 μs | 20.61 μs |
| logspace / 4,096 | 32.08 μs | 57.94 μs | 57.91 μs |
| logspace / 1,000,000 | 5464.99 μs | 8782.35 μs | 8687.50 μs |

These results show the tradeoff by size: extra native-compatible validation can cost more on small
`arange` calls, and medium `linspace` calls can be slower; no universal speedup is claimed.
All dtypes, raw samples, process ranges and benchmark/source identities are in the JSON and archive.
The first uncalibrated browser timing attempt hit the clock-resolution floor and failed with a zero
denominator. That attempt is preserved; all twelve final children were restarted after adding batch
calibration. No samples from the uncalibrated attempt enter the reported comparison.

For one warmed CPU float32 `linspace` with one million elements, tracemalloc peak was
12,001,504 bytes for baseline and 4,002,092 bytes for candidate,
with a 4,000,000-byte result alive. This measures tracked call allocations, not process RSS or WASM
heap high-water marks. The separately bounded small-template cache is warmed before tracing.

## Integration and remaining scope

Incoming main UI commits through `f180057b1fefcb5e5bec37bd2b0f8f63ba25fb8b` were preserved by fast-forward;
all affected frontend gates ran on that combined state. Numeric source hashes and native input hashes
remain the qualification boundary. Ordinary main adoption is recorded separately after local gates.
No tag, manual workflow dispatch, release, package upload or deployment is part of this iteration.
Canonical workflows do not trigger the quality gate on a plain branch push.

This remains dense NumPy CPU/Pyodide WASM. There is no native Torch or GPU execution claim.
Half neural-network/forward-AD limits, complex AD limits, output-buffer/layout restrictions,
and iteration-53 mixed-float32 numerical qualification remain. Raw Tensor constructor conventions
and the smaller legacy schedules profile retain their documented scope. WebGPU remains a separate
backend investigation, not implied by these CPU/WASM results.

## Reproduction artifacts

- Machine-readable result: `packages/stridetorch/benchmarks/results/iteration-56.json`
- Benchmark: `packages/stridetorch/benchmarks/bench_sequences.py`
- Raw harness/archive: `packages/stridetorch/benchmarks/results/iteration-56-harness.tar.gz`
- Archive bytes: 3,533,512
- Archive SHA-256: `bdd7523d4b8a943ec644b971c0ac09618eaf46e73913d027414b9dd0aa555ebd`

The raw archive preserves frozen baseline/candidate source and tests, native arrays and metadata,
CPU/Node/Chrome qualifications, wheel/sdist artifacts, frontend/catalog receipts, diagnostic pilots,
and all 12 final benchmark children. Environments, caches and third-party C++ source are excluded.

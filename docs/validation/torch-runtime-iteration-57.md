# Torch runtime iteration 57 — bounded integer range overhead

Version 0.57.0 reduces construction overhead for the ordinary Python integer
`arange(n)`, `arange(0,n)`, `arange(0,n,1)` and equivalent keyword forms,
with `0 <= n <= 16,000,000`. This improves a measured regression disclosed in
[iteration 56](torch-runtime-iteration-56.md). The broader independent-library,
core-coverage and backend objective remains active.

## Implementation and safeguards

The bounded integer case proves its scalar conversions and output size directly,
avoiding repeated generic Number parsing. Positional arity parsing no longer
constructs a temporary keyword dictionary. Dtype/device/options and output-leaf
validation remain on every call; each output owns fresh storage. Float32/64 and
integer unit indices within the browser limit require no scoped floating error
mode; half retains that scope for overflow-to-infinity storage. No value-dependent
result cache or argument-object cache is introduced.

Tensor, forward-AD, NumPy scalar, scalar-subclass, non-unit-step, shifted and
out-of-limit controls retain the existing conversion/error path. The optimization
has no effect on the global default float dtype: integer controls still infer
int64. Unsupported dtypes, invalid options, malformed overloads, input overflow,
gradient leaves inside no_grad, empty strides and mutation independence retain
the documented behavior. No API, backend or dependency is added.

## Validation

| Gate | Result |
| --- | --- |
| Full repository CPU tests, including SDK | 40,647 passed, 1 skipped |
| Node Pyodide 0.29.4 / NumPy 2.2.5 | 7,017 passed |
| Chrome 155.0.8059.82 worker, same Pyodide | 7,017 passed |
| Installed public wheel, Python 3.10.19 / NumPy 1.26.4 | 7,030 passed |
| Exported PaperLeet catalog | 78/78 modules, 342 mutants; all original browser budgets passed |
| Independent contracts in exported attention runtime | 7,017 tests / 52 suites |
| Public/private wheels, separate uninstall, fresh pickle/alias processes | Passed |
| Frontend checks, driver/release checks, local normal and Pages-profile builds | All five passed |
| Source/license identities | 59 resources across source, wheels and WASM |
| Immutable catalog payloads retained | 11,967 unchanged; 158 reachable new payloads |
| Active exports/indices | 161 identities checked |
| Canonical release policies | Seven byte-identical to `13d2782a494415ba72bc604a0ef7e4c4f875ba25` |

The new 39 tests exercise all overload forms and owned storage across supported
dtypes, half overflow/integer wrapping under all three NumPy error modes, and
requested gradient leaves created inside no_grad. Native comparison adds 280
bounded-range cases spanning empty, singleton, small, integer wrap and half
infinity boundaries, on four overload forms and all fourteen dtype requests.

The full 4,495-case native Torch 2.14.1
dataset is replayed on CPU, Node WASM and Chrome WASM. Each has 2,747 native
numeric cases / 2,761 arrays, 1,634 checked error/unsupported cases, five wide
half spans validated with the independent rational oracle, and 109 retained
platform-dependent out-of-range integer casts excluded from portable numeric
claims. Dtype/shape and integers match exactly; floating tolerances and separate
real/imaginary finite/NaN/signed-infinity checks retain iteration 56's explicit
qualification. Unsupported cases and the five rational-oracle cases are not
native numerical passes. All 1,152 native catalog input hashes were rechecked.

Package/CPU/SDK/frontend gates finish before the complete WASM catalog budget run.
All heavy validation jobs finish before formal performance timing. This avoids
mixing the measured construction improvement with validation-job contention.
Raw logs and source/test/runtime identities remain available for each gate.

## Performance

Baseline is validated 0.56.0 (`2998cc072ff2642bd800d04357d5c8c2bf17dbcd`), candidate 0.57.0.
Four fresh processes per CPU/Node/Chrome runtime, five samples per case; order
alternates by sample and runtime. Batch counts are calibrated outside timing to
at least 20 ms per variant, then shared within a case. Correctness, warmup and
GC are outside timing; result construction/destruction is included.

Candidate latency / candidate-to-baseline time ratio, with the full process-ratio
range in parentheses. Below one means faster.

| arange dtype / elements | CPU | Node WASM | Chrome WASM |
| --- | --- | --- | --- |
| float32 / 16 | 1.39 μs / 0.381× (0.375–0.382) | 3.43 μs / 0.390× (0.388–0.391) | 3.43 μs / 0.398× (0.390–0.415) |
| float32 / 4,096 | 2.47 μs / 0.516× (0.515–0.518) | 4.68 μs / 0.468× (0.462–0.468) | 4.63 μs / 0.471× (0.468–0.476) |
| float32 / 1,000,000 | 251.38 μs / 0.989× (0.979–0.992) | 273.78 μs / 0.979× (0.978–0.980) | 276.17 μs / 0.981× (0.978–0.983) |
| int64 / 16 | 1.49 μs / 0.314× (0.310–0.316) | 3.73 μs / 0.324× (0.323–0.326) | 3.72 μs / 0.335× (0.328–0.340) |
| int64 / 4,096 | 1.94 μs / 0.367× (0.357–0.369) | 5.07 μs / 0.391× (0.390–0.394) | 5.11 μs / 0.405× (0.404–0.405) |
| int64 / 1,000,000 | 108.52 μs / 0.971× (0.967–0.973) | 314.81 μs / 0.976× (0.972–0.982) | 317.97 μs / 0.974× (0.971–0.976) |

The benchmark also records the unchanged linspace/logspace controls and all
other supported dtype/size cases. Large ranges are dominated by array generation;
no universal speedup or GPU throughput is claimed. The JSON and raw archive
contain all 29 rows in each of twelve children, raw samples, calibrated batch
sizes and exact baseline/candidate/benchmark hashes. Profiling was used to locate
parsing/context overhead; reported speed ratios use unprofiled timings.
The retained tracemalloc linspace measurement is an unchanged allocation control,
not a new memory reduction claim and not process RSS or WASM heap usage.

## Integration and limits

Incoming main through `2998cc072ff2642bd800d04357d5c8c2bf17dbcd` is preserved, with frontend gates on
the final combined checkout. Naming remains StrideTorch pending a separate choice.
No tag, workflow dispatch, release, package upload or deployment is initiated;
ordinary main adoption is recorded separately after the local gates.

Existing dense CPU/Pyodide WASM limits remain: no native Torch/GPU execution,
half neural-network/forward-AD and complex-AD limits, output-buffer/layout limits,
the smaller legacy profile, and iteration-53 mixed-float32 qualification.
WebGPU remains a separate backend investigation. This version improves a common
range-construction path while preserving the established compatibility boundary.

## Reproduction

- Result: `packages/stridetorch/benchmarks/results/iteration-57.json`
- Benchmark: `packages/stridetorch/benchmarks/bench_sequences.py`
- Raw archive: `packages/stridetorch/benchmarks/results/iteration-57-harness.tar.gz`
- Archive bytes: 4,687,936
- Archive SHA-256: `6d3cd7d799355b665a340adcf2e026324c9f4e1c7e3d3299f257a3cb74eeddf3`

The archive contains frozen baseline/candidate code, tests, native arrays,
CPU/Node/Chrome oracle/contract results, installed package evidence, frontend and
catalog receipts, profiling/pilot data and all twelve final benchmark children.
Environment directories, caches and third-party implementation source are excluded.

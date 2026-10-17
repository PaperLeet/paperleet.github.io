# Torch runtime iteration 63 — robust determinants under NumPy policies

Version **0.63.0** fixes false floating-point failures in determinant evaluation
and exact 1×1 determinant values/gradients at extreme scales. Final integration
base is `11d613a3465d92966af6fb606a497b2a0ac3801b`; the full CPU/frontend/catalog baseline is
`789f78bc3ca0d7a37ae249418ecb68de28039122`. 8 affected checks at `49a8ddd31cfc7743b34e419acc48ef0fa61984ee` plus driver/policy/two-build gates; 6 affected checks at `11d613a3465d92966af6fb606a497b2a0ac3801b` plus driver/policy/two-build gates

## Actual failures and correction

The prior 0.62 implementation raises `FloatingPointError` for finite 32×32,
64×64 and 128×128 identity determinants under `numpy.seterr(all='raise')` on the
tested CPU NumPy 2.5.4 build. A frozen 162-case diagnostic across nine linalg
APIs, two real dtypes and nine sizes identifies determinant failures; the same
candidate diagnostic has no errors. This is a runtime observation, not a claim
that every NumPy/BLAS build has the defect.

On runtimes that report IEEE error flags, determinant factorization now runs
inside a local NumPy ignore context. Valid finite Torch inputs no longer depend
on the caller's NumPy arithmetic exception/callback policy. The policy and callback
are restored. The WASM runtime that cannot report those flags avoids that context.
Existing dtype, square-matrix, finite-input and output-buffer guards remain in
place. Genuine determinant overflow/underflow keeps its IEEE value rather than
becoming a Python exception.

For 1×1 matrices, the determinant is the stored element and its differential is
the incoming seed. Copying that value avoids logarithm/exponential rounding.
Returning the seed view avoids multiplying by a huge/tiny determinant and then
dividing it back out. Native reference confirms the corrected behavior: the old
maximum-float32 matrix with seed 2 produced Inf instead of 2; the old smallest
float64 subnormal could produce NaN/Inf instead of the seed. Negative zero is
preserved. Raw VJP returns native seed-view strides/aliasing, including strided
batches; leaf `.grad` still owns its buffer. Empty raw VJPs preserve native input
layout. Input/output saved-value mutation guards remain in force.

No other determinant cofactor contract, log API, JVP boundary or runtime dependency
is changed. General singular/extreme matrices retain the existing cofactor path.
Real first-order gradients remain the scope; det JVP and complex det remain
unsupported. Existing broader limits are recorded in the
[0.62 report](torch-runtime-iteration-62.md).

## Verification

- **41,084 CPU passes, 1 skip**, exact source hashes checked before and after.
- Actual Node/Pyodide and Chrome Worker: **7,454 independent contracts each**,
  all **55 suites**. Python 3.10.19 / NumPy 1.26.4 installed public wheel:
  **7,467 passes**, native Torch absent.
- **114 new independent SDK cases** cover exact singleton polynomial values and
  slopes, negative zero, subnormals/maximum finite inputs, Inf/NaN seeds, NumPy
  raise/warn/call policies, restored callbacks, seed aliases versus owned leaf
  gradients, empty/factory layouts and saved-value mutation failures.
- Original 600-case qualification remains: 480 native log cases, 54 ordinary
  native det cases, 48 separate legacy det cases and 18 retained restriction
  cases, with 1,764 numerical arrays and 360 None-gradient fields per runtime.
- Additional **398 frozen singleton cases / 796 exact inputs and seeds** compare
  **1,194 arrays** with native Torch 2.14.1. C/F/sliced data and seeds, scalar and
  batched cases, empty NumPy/factory inputs, output ownership and raw seed aliases
  are checked. Values, NaN/Inf masks and non-NaN sign bits are exact. The 118
  equivalent positive-singleton stored-gradient layouts retain the documented
  allocation rule; raw VJP layouts are checked exactly.
- Both qualifiers run on CPU, Node, Chrome and the isolated minimum environment.
  Native reference git is `5c4886908584029761b579af026dcfb627c84070`. Legacy
  cofactor/restriction cases are not reported as native passes.
- Public wheel/sdist and vendored wheel match all 59 Python modules plus two
  license resources. Coexistence, uninstalling either owner, alias restoration,
  fresh-process model/activation serialization and native Torch isolation pass.
- All **78 exported modules and 342 mutants** pass original browser budgets.
  The embedded independent contracts include all 55 suites. All 1,152 native
  catalog input hashes and 12,915 earlier immutable payloads are unchanged;
  158 new assets are reachable and 161 active export/index hashes are recorded.
- Full frontend checks and both build profiles pass at the recorded baseline;
  later disjoint UI changes, if any, have explicit staged qualification above.
- Seven release-policy files are byte-identical to canonical
  `13d2782a494415ba72bc604a0ef7e4c4f875ba25`. No CI trigger, tag, workflow dispatch,
  release, package upload or deployment is initiated.

## Performance and cost against actual 0.62

Twelve fresh children, four CPU/Node/Chrome each, **100 matched workflows per
child**, five alternating samples with common calibrated batches of at least
20 ms. Both variants invoke actual 0.62/0.63 determinant/log APIs. Tables give
median child ratios and ranges; below 1 is faster. Inputs/conversions/seeds
are outside timing; dispatch, graphs, kernels, weighted backward, clearing and
result destruction are inside. Correctness/budget runs finish before timing.
CPU thread caps are two; no host-wide isolation, cold-start, native-Torch or GPU
throughput claim is made.

1×1 determinant workflows:

| Runtime | Operation | Dtype | Shape | Mode | Candidate / 0.62 | Four-child range |
|---|---|---|---|---|---:|---:|
| cpu | det | float32 | [1, 1] | forward | 0.590 | 0.589–0.597 |
| cpu | det | float32 | [1, 1] | full_step | 0.613 | 0.607–0.625 |
| cpu | det | float64 | [1, 1] | forward | 0.644 | 0.640–0.660 |
| cpu | det | float64 | [1, 1] | full_step | 0.633 | 0.625–0.641 |
| node | det | float32 | [1, 1] | forward | 0.640 | 0.625–0.647 |
| node | det | float32 | [1, 1] | full_step | 0.649 | 0.642–0.660 |
| node | det | float64 | [1, 1] | forward | 0.716 | 0.707–0.723 |
| node | det | float64 | [1, 1] | full_step | 0.676 | 0.671–0.677 |
| chrome | det | float32 | [1, 1] | forward | 0.635 | 0.620–0.651 |
| chrome | det | float32 | [1, 1] | full_step | 0.645 | 0.632–0.661 |
| chrome | det | float64 | [1, 1] | forward | 0.727 | 0.720–0.732 |
| chrome | det | float64 | [1, 1] | full_step | 0.668 | 0.662–0.679 |

The CPU error-policy context has a cost for ordinary 2×2 forward-only calls.
These costs are retained explicitly rather than hidden by singleton gains:

| Runtime | Operation | Dtype | Shape | Mode | Candidate / 0.62 | Four-child range |
|---|---|---|---|---|---:|---:|
| cpu | det | float32 | [2, 2] | forward | 1.149 | 1.147–1.164 |
| cpu | det | float64 | [2, 2] | forward | 1.155 | 1.143–1.193 |
| node | det | float32 | [2, 2] | forward | 1.022 | 1.008–1.027 |
| node | det | float64 | [2, 2] | forward | 1.016 | 1.003–1.026 |
| chrome | det | float32 | [2, 2] | forward | 1.012 | 0.997–1.034 |
| chrome | det | float64 | [2, 2] | forward | 1.019 | 1.006–1.029 |

64×64 determinant workflows:

| Runtime | Operation | Dtype | Shape | Mode | Candidate / 0.62 | Four-child range |
|---|---|---|---|---|---:|---:|
| cpu | det | float32 | [64, 64] | forward | 0.971 | 0.947–0.984 |
| cpu | det | float32 | [64, 64] | full_step | 0.994 | 0.987–0.996 |
| cpu | det | float64 | [64, 64] | forward | 0.974 | 0.943–0.984 |
| cpu | det | float64 | [64, 64] | full_step | 0.993 | 0.980–1.012 |
| node | det | float32 | [64, 64] | forward | 1.000 | 0.953–1.008 |
| node | det | float32 | [64, 64] | full_step | 0.997 | 0.991–1.008 |
| node | det | float64 | [64, 64] | forward | 1.004 | 0.996–1.014 |
| node | det | float64 | [64, 64] | full_step | 0.995 | 0.975–1.000 |
| chrome | det | float32 | [64, 64] | forward | 1.005 | 1.000–1.009 |
| chrome | det | float32 | [64, 64] | full_step | 1.007 | 1.000–1.007 |
| chrome | det | float64 | [64, 64] | forward | 1.005 | 1.000–1.019 |
| chrome | det | float64 | [64, 64] | full_step | 1.004 | 0.996–1.014 |

Graph controls:

| Runtime | Operation | Dtype | Shape | Mode | Candidate / 0.62 | Four-child range |
|---|---|---|---|---|---:|---:|
| cpu | graph | float32 | [1] | full_step | 0.994 | 0.986–0.999 |
| cpu | graph | float32 | [32] | full_step | 1.000 | 0.985–1.047 |
| cpu | graph | float64 | [1] | full_step | 0.987 | 0.982–1.022 |
| cpu | graph | float64 | [32] | full_step | 0.990 | 0.928–1.018 |
| node | graph | float32 | [1] | full_step | 0.998 | 0.994–1.025 |
| node | graph | float32 | [32] | full_step | 1.003 | 0.998–1.024 |
| node | graph | float64 | [1] | full_step | 1.008 | 0.995–1.013 |
| node | graph | float64 | [32] | full_step | 1.007 | 1.006–1.016 |
| chrome | graph | float32 | [1] | full_step | 0.994 | 0.988–1.012 |
| chrome | graph | float32 | [32] | full_step | 1.000 | 0.992–1.012 |
| chrome | graph | float64 | [1] | full_step | 1.004 | 0.996–1.013 |
| chrome | graph | float64 | [32] | full_step | 1.002 | 0.991–1.017 |

Largest matched ratios, including losses:

| Runtime | Operation | Dtype | Shape | Mode | Candidate / 0.62 | Four-child range |
|---|---|---|---|---|---:|---:|
| cpu | det | float64 | [2, 2] | forward | 1.155 | 1.143–1.193 |
| cpu | det | float64 | [8, 8] | forward | 1.149 | 1.140–1.166 |
| cpu | det | float32 | [2, 2] | forward | 1.149 | 1.147–1.164 |
| cpu | det | float32 | [8, 8] | forward | 1.149 | 1.143–1.330 |
| cpu | det | float64 | [4, 8, 8] | forward | 1.127 | 1.123–1.137 |
| cpu | det | float32 | [4, 8, 8] | forward | 1.122 | 1.114–1.129 |
| cpu | det | float32 | [2, 2] | full_step | 1.045 | 1.038–1.049 |
| cpu | det | float64 | [4, 8, 8] | full_step | 1.036 | 1.028–1.040 |

All 300 runtime/workflow aggregate rows, 1,200 child workflow results, samples,
calibration values and source identities are retained. Float32 benchmark gradient
checks keep the existing separately labeled 1e-5 normwise bound; native/singleton
correctness qualification does not relax its checks.

## Allocation and reproduction

Warm incremental CPU tracemalloc allocations are recorded for 24 ordinary
64×64/4×32×32 workflows. `warm-memory.py` primes both actual API paths; inputs
and seeds predate tracing. Forward retained/peak and backward peak are separate,
not total tape bytes, process RSS, LAPACK workspace or WASM heap size. Cold-first
diagnostics remain separate. Full allocation/footprint tables are in the machine
report; no new runtime dependency or GPU path is introduced.

[Machine report](../../packages/stridetorch/benchmarks/results/iteration-63.json)
and [raw harness](../../packages/stridetorch/benchmarks/results/iteration-63-harness.tar.gz).
Archive: 5,218,350 bytes; SHA-256 `6a4f4ec2cfff232c66360d88b20f25f734e7b383c82b4630406ed4aba742e3ba`. Source snapshots, frozen
native inputs/results, policy diagnostics, programs, plans, package artifacts,
receipts, development failures and samples are included. The public benchmark
is `packages/stridetorch/benchmarks/bench_determinants.py`.

The broader independent-library/core-coverage/backend objective remains active.

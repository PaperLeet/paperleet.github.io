# Torch runtime iteration 64 — determinant forward AD

Version **0.64.0** adds real float32/64 determinant JVPs through `torch.det`,
`torch.linalg.det` and `Tensor.det()`. Final integration baseline is
`18028df1389fa621e1aeb3120c937cc580a5af75`; full CPU/frontend/catalog validation baseline is
`7cd7c4f8c3dc74eb5be96dfcec0922249b991c40`. 8 affected checks plus driver/policy/two builds at `18028df1389fa621e1aeb3120c937cc580a5af75`

## Missing API and numerical behavior

The prior 0.63 implementation rejects all 52 initial determinant JVP probes.
The new implementation computes the native saved-factor directional rule and
supports reverse propagation through both returned channels. Normal, negative,
singular, mixed batched, strided, empty and extreme finite matrices are covered.
Complex AD, output buffers and nonfinite matrix inputs remain unsupported.
Directions and gradient seeds retain the separately tested IEEE behavior.

The native reference is **PyTorch 2.14.1**, git
`5c4886908584029761b579af026dcfb627c84070`. Its
[pinned forward-AD implementation](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/csrc/autograd/FunctionsManual.cpp#L4419)
uses saved LU factors, replacing exactly zero LU diagonal pivots by the dtype's
epsilon. Reverse differentiation retains the saved-factor behavior; it does
not differentiate a newly recomputed factorization. This compatibility target
is explicit and does not promise arbitrary higher derivatives or parity with
future upstream changes.

Consequently, native singular determinant JVPs can be zero even when the
polynomial differential is nonzero. Inside JVP, the primal and tangent reverse
paths follow this pinned native convention. Ordinary `det` backward keeps its
existing polynomial-cofactor extension for singular/extreme values; the old
48-case legacy qualification is retained separately. The 0.63 exact singleton
ordinary value/VJP and floating-point-policy fixes remain in force.

The returned primal is a metadata alias of the private determinant result.
Retaining its gradient or registering its hook does not intercept the tangent's
private factor path. Primal storage mutations are still guarded. Mutating the
direction or returned tangent follows the tested native behavior. A dedicated
48-case mutation probe and independent SDK tests cover all requires-grad pairs
with a differentiable output and both channels.

## Computation and lifetime

The implementation uses NumPy's matrix solve for ordinary batches and invokes
the existing pivoted-LU fallback only for singular batches. Determinant JVP opts
into epsilon pivots; log determinant singular IEEE handling is unchanged.
Float32 singleton reverse solves preserve native reciprocal precision instead
of accepting NumPy's promoted solve behavior at subnormal scales.

One immutable matrix snapshot is shared between the two reverse paths when
gradients need matrix values. Forward-only/no-grad JVP avoids that snapshot.
Only the solve trace is retained; the full temporary solution is released.
Two independent reverse edges allow the existing graph engine to skip the
unrequested direction solve. The returned-primal alias keeps hooks and graph
lifetime behavior distinct from the private determinant value. No new runtime
dependency or GPU path is introduced.

## Verification

- **41,253 CPU tests pass, 1 skip**; exact source identity is checked before and
  after the run.
- **7,623 contracts / 56 suites** pass independently on Node/Pyodide and Chrome
  Worker. The Python 3.10.19 / NumPy 1.26.4 installed wheel passes **7,636 tests**
  with native Torch absent.
- **169 new independent SDK tests** cover analytic 2×2 polynomial oracles,
  native singleton/epsilon/IEEE rules, singular JVP versus ordinary cofactor
  gradients, three aliases, C/F/sliced inputs, mutation guards, returned-primal
  hooks, graph pruning, snapshot lifetime, no-grad empties and error policies.
- **696 new frozen native controls** compare **3,132 numerical arrays and
  3,132 None-gradient/hook fields** on CPU, Node, Chrome and the minimum wheel.
  They include 58 exact matrix/direction/seed groups (174 arrays), three layouts,
  all four requires-grad pairs, mixed singular batches, 32×32 matrices, Inf/NaN
  directions, raw/stored gradients and both primal/tangent reverse paths.
- NaN/+Inf/-Inf masks are exact. Float64 finite values use componentwise
  `rtol=3e-12, atol=0`; float32 finite values use `rtol=2e-5, atol=0`, with a
  per-matrix max-norm bound `1e-5` for float32 matrix gradients at size >=8.
  Eight initial strict componentwise differences involved near-zero entries in
  32×32 float32 gradients; the retained float64 diagnostic has scaled max error
  below 6.5e-7. No IEEE mask checks are relaxed. Singleton output zero signs
  are checked exactly. The **48** equivalent positive singleton stored-gradient
  layouts follow the existing owned-buffer contract; raw layouts are exact.
- The earlier **600-case determinant/log qualification** and **398 singleton
  controls** still pass on all four environments. Native, legacy-cofactor and
  input-restriction cases remain separately labeled.
- Public wheel/sdist, vendored wheel, coinstallation/uninstall isolation,
  native-Torch alias restoration and fresh-process serialization are checked.
  Both package forms contain the same 59 Python modules and two license files.
- All **78 exported modules and 342 mutants** pass the original browser budgets;
  the embedded independent SDK run includes all 56 suites. All 1,152 native
  catalog input hashes and **13,073 prior immutable payloads** are unchanged;
  158 new reachable assets and 161 active export/index identities are recorded.
- Full frontend checks and both build profiles pass at the recorded baseline.
  Any later disjoint UI changes are qualified explicitly above.
- Seven release-policy files remain byte-identical to canonical
  `13d2782a494415ba72bc604a0ef7e4c4f875ba25`. No CI trigger, tag, workflow dispatch,
  release, package upload or deployment is initiated.

## Performance with explicit comparators

Twelve fresh children (four CPU, four Node and four Chrome), with two CPU
threads capped per supported backend, run five alternating samples with common
calibrated batches of at least 20 ms. Correctness/budget runs finish before
formal timing. This is host measurement without machine-wide isolation.

The regression comparator invokes actual 0.63 APIs for **100 matched workflows
per child** (1,200 workflow results): real/complex determinant/log forward,
real weighted reverse and graph controls. All 300 aggregate rows and original
samples are retained. The largest median ratios, including losses, are:

| Runtime | Operation | Dtype | Shape | Mode | 0.64 / actual 0.63 | Four-child range |
|---|---|---|---|---|---:|---:|
| chrome | det | float32 | [8, 8] | full_step | 1.023 | 1.003–1.074 |
| chrome | det | float64 | [2, 2] | forward | 1.021 | 1.015–1.024 |
| node | det | float64 | [32, 32] | full_step | 1.018 | 1.001–1.027 |
| node | det | float64 | [1, 1] | forward | 1.017 | 1.002–1.028 |
| cpu | logdet | float32 | [1, 1] | full_step | 1.016 | 0.997–1.036 |
| chrome | det | float64 | [8, 8] | full_step | 1.015 | 1.005–1.040 |
| cpu | det | float32 | [4, 32, 32] | forward | 1.014 | 1.011–1.016 |
| node | slogdet | float64 | [8, 8] | full_step | 1.014 | 0.975–1.029 |
| cpu | slogdet | float32 | [1, 1] | full_step | 1.014 | 0.994–1.044 |
| node | det | float32 | [8, 8] | forward | 1.014 | 1.008–1.022 |

Graph controls:

| Runtime | Operation | Dtype | Shape | Mode | 0.64 / actual 0.63 | Four-child range |
|---|---|---|---|---|---:|---:|
| cpu | graph | float32 | [1] | full_step | 1.003 | 0.994–1.021 |
| cpu | graph | float32 | [32] | full_step | 1.003 | 1.000–1.006 |
| cpu | graph | float64 | [1] | full_step | 1.004 | 0.999–1.017 |
| cpu | graph | float64 | [32] | full_step | 1.003 | 0.988–1.017 |
| node | graph | float32 | [1] | full_step | 1.004 | 1.000–1.015 |
| node | graph | float32 | [32] | full_step | 0.996 | 0.984–1.001 |
| node | graph | float64 | [1] | full_step | 0.994 | 0.989–1.007 |
| node | graph | float64 | [32] | full_step | 1.001 | 0.995–1.015 |
| chrome | graph | float32 | [1] | full_step | 1.000 | 0.996–1.004 |
| chrome | graph | float32 | [32] | full_step | 1.002 | 0.992–1.004 |
| chrome | graph | float64 | [1] | full_step | 1.002 | 0.997–1.021 |
| chrome | graph | float64 | [32] | full_step | 1.002 | 0.983–1.021 |

The new-feature comparator has **16 additional workflows per child** (192
results, 48 aggregates). Since 0.63 has no determinant JVP API, its comparator
is explicitly the existing-API composition
`det(A) * trace(solve(A.detach(), E))`, tested only on finite nonsingular
matrices. It checks values and weighted matrix/direction gradients, not alias,
hook or singular equivalence. Those differences are covered by native controls.
Below 1 means the new direct API is faster than that composition:

| Runtime | Dtype | Shape | Mode | Direct JVP / composition | Four-child range |
|---|---|---|---|---:|---:|
| cpu | float32 | [1, 1] | forward | 1.014 | 1.009–1.017 |
| cpu | float32 | [1, 1] | full_step | 0.805 | 0.750–0.822 |
| cpu | float32 | [2, 2] | forward | 1.017 | 1.008–1.031 |
| cpu | float32 | [2, 2] | full_step | 0.802 | 0.794–0.809 |
| cpu | float32 | [64, 64] | forward | 0.982 | 0.979–0.996 |
| cpu | float32 | [64, 64] | full_step | 0.914 | 0.912–0.921 |
| cpu | float32 | [4, 32, 32] | forward | 0.972 | 0.964–0.987 |
| cpu | float32 | [4, 32, 32] | full_step | 0.942 | 0.925–0.951 |
| cpu | float64 | [1, 1] | forward | 1.032 | 1.004–1.046 |
| cpu | float64 | [1, 1] | full_step | 0.815 | 0.811–0.837 |
| cpu | float64 | [2, 2] | forward | 1.010 | 0.994–1.022 |
| cpu | float64 | [2, 2] | full_step | 0.801 | 0.769–0.835 |
| cpu | float64 | [64, 64] | forward | 0.986 | 0.972–0.998 |
| cpu | float64 | [64, 64] | full_step | 0.919 | 0.909–0.926 |
| cpu | float64 | [4, 32, 32] | forward | 0.973 | 0.955–0.983 |
| cpu | float64 | [4, 32, 32] | full_step | 0.917 | 0.811–0.934 |
| node | float32 | [1, 1] | forward | 1.009 | 0.979–1.051 |
| node | float32 | [1, 1] | full_step | 0.825 | 0.821–0.834 |
| node | float32 | [2, 2] | forward | 0.992 | 0.983–1.002 |
| node | float32 | [2, 2] | full_step | 0.809 | 0.802–0.834 |
| node | float32 | [64, 64] | forward | 0.998 | 0.973–1.006 |
| node | float32 | [64, 64] | full_step | 0.954 | 0.951–0.957 |
| node | float32 | [4, 32, 32] | forward | 0.991 | 0.983–1.045 |
| node | float32 | [4, 32, 32] | full_step | 0.952 | 0.937–0.963 |
| node | float64 | [1, 1] | forward | 0.996 | 0.980–1.007 |
| node | float64 | [1, 1] | full_step | 0.848 | 0.825–0.933 |
| node | float64 | [2, 2] | forward | 0.983 | 0.980–0.996 |
| node | float64 | [2, 2] | full_step | 0.812 | 0.806–0.818 |
| node | float64 | [64, 64] | forward | 0.997 | 0.989–1.016 |
| node | float64 | [64, 64] | full_step | 0.950 | 0.950–0.957 |
| node | float64 | [4, 32, 32] | forward | 0.990 | 0.987–0.991 |
| node | float64 | [4, 32, 32] | full_step | 0.949 | 0.934–0.961 |
| chrome | float32 | [1, 1] | forward | 0.999 | 0.986–1.017 |
| chrome | float32 | [1, 1] | full_step | 0.820 | 0.813–0.826 |
| chrome | float32 | [2, 2] | forward | 0.984 | 0.977–0.998 |
| chrome | float32 | [2, 2] | full_step | 0.818 | 0.801–0.828 |
| chrome | float32 | [64, 64] | forward | 1.006 | 0.989–1.016 |
| chrome | float32 | [64, 64] | full_step | 0.953 | 0.916–0.957 |
| chrome | float32 | [4, 32, 32] | forward | 0.987 | 0.982–1.000 |
| chrome | float32 | [4, 32, 32] | full_step | 0.943 | 0.939–0.950 |
| chrome | float64 | [1, 1] | forward | 0.991 | 0.976–1.006 |
| chrome | float64 | [1, 1] | full_step | 0.835 | 0.831–0.844 |
| chrome | float64 | [2, 2] | forward | 0.986 | 0.981–1.003 |
| chrome | float64 | [2, 2] | full_step | 0.813 | 0.810–0.826 |
| chrome | float64 | [64, 64] | forward | 0.995 | 0.992–1.008 |
| chrome | float64 | [64, 64] | full_step | 0.950 | 0.941–0.960 |
| chrome | float64 | [4, 32, 32] | forward | 0.986 | 0.982–0.993 |
| chrome | float64 | [4, 32, 32] | full_step | 0.941 | 0.931–0.948 |

Inputs, conversions, directions and seeds precede timing. Dispatch/transform,
graph construction, kernels, optional weighted backward, clearing and result
destruction are timed. No native-Torch, GPU, cold-start or universal speedup
claim is made.

## Allocation and reproduction

The 24 ordinary determinant/log allocation diagnostics retain separate warm and
cold-first results. Eight additional warm CPU tracemalloc rows measure complete
weighted JVP steps against the composition. Inputs/directions/seeds predate
tracing; tracked retained allocations include newly stored leaf gradients.
The direct API's source snapshot and the composition's borrowed guarded input
are different lifetime contracts. These are not total tape bytes, process RSS,
LAPACK workspace or WASM heap measurements. In this one warm diagnostic,
tracked step peaks are 7–36% higher for the direct API, while retained bytes
are roughly 1.3–1.4 KB lower. The required immutable source snapshot has a
visible allocation cost; this iteration does not claim a universal RAM saving.

| Dtype | Shape | Composition step peak bytes | Direct JVP step peak bytes | Direct / composition |
|---|---|---:|---:|---:|
| float32 | [64, 64] | 211,347 | 225,746 | 1.068 |
| float32 | [4, 32, 32] | 162,432 | 176,415 | 1.086 |
| float64 | [64, 64] | 178,119 | 208,966 | 1.173 |
| float64 | [4, 32, 32] | 129,488 | 175,823 | 1.358 |

[Machine report](../../packages/stridetorch/benchmarks/results/iteration-64.json)
and [raw harness](../../packages/stridetorch/benchmarks/results/iteration-64-harness.tar.gz)
contain source/test snapshots, frozen inputs/native outputs, programs, failed
development pilots, source identities, packages, receipts and all samples.
Archive: 6,296,457 bytes; SHA-256 `d824de2bd648fc14f0a8fd077653b1cb1ab78e5cc41cd50d0b2670ec1e5eefd5`.

CPU/WASM is the only backend changed this iteration. The prior WebGPU design
assessment remains applicable; no GPU prototype or GPU performance is claimed.
The wider independent-library/core-coverage/backend objective remains active.

## Main adoption

Validated source `0a5da0c1b94869e901a1427170ec17ddc5323b41` was adopted on remote main by ordinary non-force push.
The remote SHA matched exactly, with zero observed Actions runs for that SHA.
All 61 source/license resources, 161 active exports, 13,231 immutable payloads
and seven canonical release-policy files were rechecked. A separate proof commit
records this observation; its final main identity is checked outside the repository.
No tag, workflow dispatch, release, PyPI upload or deployment was initiated.
The broader library efficiency/core-coverage/backend objective remains active.

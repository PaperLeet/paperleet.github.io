# Torch runtime iteration 65 — inverse APIs and saved-value guards

Version **0.65.0** adds `torch.inverse`, `Tensor.inverse()`, native-style
`torch.linalg.inv(A=...)`, real inverse forward AD and complex inverse forward
calculation. Existing `torch.linalg.inv(input=...)` calls remain supported.
Final integration baseline is `11c68b0d8ecbd7a45648a8b03c0be357f33bf594`; full CPU/frontend/catalog
validation baseline is `ec6ed5ba6896053069393271328f02b11dfc18d0`. 12 affected checks plus driver/policy/two builds at `11c68b0d8ecbd7a45648a8b03c0be357f33bf594`

## Behavior and numerical scope

Real float32/64 inverse supports ordinary reverse AD and `torch.func.jvp`.
Complex64/128 supports forward calculation; the existing constructor boundary
continues to reject complex AD. Singular matrices raise `RuntimeError`.
Nonfinite matrix inputs and `out` buffers remain unsupported. Empty matrices,
empty batches, C/F/sliced inputs, mixed batches and finite extreme values are
covered. Directions and reverse seeds retain the separately tested IEEE rules.

Inverse outputs now have native column-major final axes instead of the previous
row-major result. Singleton inverses use a dtype-preserving reciprocal, with an
explicit zero check, avoiding observed complex LAPACK overflow/subnormal errors.
For larger matrices, inversion of the transposed input exposes the desired
transposed output buffer directly. The final JVP product similarly creates its
column-major result without a second full-matrix layout copy.

The native reference is **PyTorch 2.14.1**, git
`5c4886908584029761b579af026dcfb627c84070`. Its
[pinned inverse differentiation rules](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/tools/autograd/derivatives.yaml#L831)
specify a right-associated inverse VJP and a JVP with two matrix products.
The implementation preserves these associations and saves the computed inverse
for reverse propagation instead of recomputing a factorization. This is a
first-order compatibility target, including the tested reverse paths through
the JVP; arbitrary higher derivatives are not promised.

The original input is not retained for inverse backward. Its later mutation is
allowed; mutation of the saved inverse is guarded. A direction whose feature
strides differ from the primal is copied when the primal reverse path needs it,
following measured native forward-AD behavior. Otherwise the needed direction
storage is guarded. The returned primal is a separate metadata alias, so its
hook and retained gradient do not intercept the tangent's private inverse path.
Independent edges let the graph engine prune unrequested reverse branches.

Matrix multiplication also stops guarding outputs and operands that its active
derivatives never read. It still guards each saved operand needed by an active
derivative, including shared operands. Of 105 native guard controls, the prior
version falsely rejected 55 permitted mutation cases; the new version matches
all 105. These are observable compatibility fixes as well as graph improvements.

## Verification

- **41,596 CPU tests pass, 1 skip** with exact source hashes checked before and
  after the run. **343 new SDK tests** include independent scalar/2×2 derivative
  formulas, aliases, both reverse paths, layouts, dtype extremes, mutations,
  NumPy error policy, hooks, saved-source lifetime and shared operands.
- **7,966 contracts / 58 suites** pass independently on Node/Pyodide and Chrome
  Worker. The isolated Python 3.10.19 / NumPy 1.26.4 installed wheel passes
  **7,979 tests**, with native Torch absent.
- The inverse frozen corpus has **96 input groups / 224 arrays**, expanded to
  **864 contexts**: 528 successful real JVP cases, 78 successful complex forwards
  and 258 native singular rejections. It compares **2,454 numerical arrays** and
  **2,376 None-gradient/hook fields** on CPU, Node, Chrome and the minimum wheel.
- Another **1,080 mutation/layout cases**, **105 matmul guard cases** and **120
  API calls** pass on all four environments. The API set separates 96 native
  canonical successes from 24 intentionally preserved legacy `input=` calls.
- IEEE NaN/+Inf/-Inf masks, dtype and geometry are exact. Matrices of size >=8
  use per-matrix/component max-norm bounds of `1e-5` for float32/complex64 and
  `3e-12` for float64/complex128. Smaller finite values use componentwise
  `rtol=3e-5` or `3e-12`, `atol=0`. The 32 equivalent positive singleton
  stored-gradient layouts follow the existing owned-buffer contract; raw and
  meaningful nonsingleton strides remain exact.
- The prior **600 determinant/log**, **398 singleton** and **696 determinant
  JVP** controls remain passing on all four environments. Native, legacy
  singular cofactor and unsupported-input expectations remain separately named.
- One old root test expected complex inverse rejection. It now checks complex
  forward values and column-major strides against native Torch and retains
  singular/shape/out and complex-AD-boundary checks. The initial failed run is
  preserved; the final full suite passes.
- Public wheel/sdist, vendored wheel, coinstallation/uninstall isolation,
  native-Torch alias restoration and fresh-process serialization pass. Both
  package forms contain the same 59 Python modules and LICENSE/NOTICE files.
- All **78 exported modules and 342 mutants** pass the original browser budgets,
  with all 58 SDK suites embedded. The **1,152 native catalog input hashes** and
  **13,231 prior immutable payloads** are unchanged. This iteration adds 158
  reachable assets and records 161 active export/index hashes.
- Full frontend checks and both build profiles pass. All 18 public/fixture
  runtime assets, 13 lock-pinned package wheels and five Node core identities
  are revalidated. Pyodide is 0.29.4 / NumPy 2.2.5; development CPU uses Python
  3.12.11 / NumPy 2.5.4 on Apple M4 arm64. Exact host metadata is archived.
- Seven release-policy files remain byte-identical to canonical
  `13d2782a494415ba72bc604a0ef7e4c4f875ba25`. No CI trigger, tag, workflow dispatch,
  release, package upload or deployment is initiated.

## Performance against actual 0.64

Twelve fresh children, four each CPU/Node/Chrome, alternate runtime order and
run five alternating common calibrated batches of at least 20 ms. Supported
CPU thread caps are two. Owned correctness/build jobs finish before formal
timing; per-child host load averages are retained. This is host measurement
without machine-wide isolation, and unrelated host work may run concurrently.

The original regression program has **100 matched actual 0.64 API workflows
per child** (1,200 results / 300 aggregates): real/complex determinant and log
forward, weighted reverse and graph controls. The largest median ratios,
including losses, are:

| Runtime | Operation | Dtype | Shape | Mode | Candidate / comparator | Four-child range |
|---|---|---|---|---|---:|---:|
| cpu | slogdet | float64 | [8, 8] | forward | 1.023 | 0.987–1.059 |
| chrome | slogdet | complex64 | [2, 2] | forward | 1.021 | 1.007–1.035 |
| chrome | det | float64 | [4, 32, 32] | full_step | 1.020 | 1.005–1.031 |
| chrome | logdet | complex128 | [2, 2] | forward | 1.018 | 0.997–1.036 |
| chrome | det | float32 | [8, 8] | full_step | 1.018 | 1.011–1.031 |
| chrome | det | float32 | [2, 2] | full_step | 1.014 | 0.984–1.032 |
| chrome | slogdet | float32 | [2, 2] | full_step | 1.014 | 1.000–1.026 |
| cpu | logdet | float64 | [4, 32, 32] | full_step | 1.012 | 1.000–1.020 |
| chrome | slogdet | float64 | [32, 32] | forward | 1.012 | 1.005–1.024 |
| cpu | logdet | float32 | [4, 32, 32] | full_step | 1.011 | 1.000–1.015 |

An additional **44 workflows per child** (528 results / 132 aggregates) comprise
16 actual prior real-inverse workflows, eight actual matmul controls, 16 new
inverse JVPs versus explicit old-API composition, and four candidate-only complex
inverse forwards. All samples and losses are retained. Ratios below one mean
less measured time. Existing inverse full steps:

| Runtime | Operation | Dtype | Shape | Mode | Candidate / comparator | Four-child range |
|---|---|---|---|---|---:|---:|
| cpu | inverse | float32 | [1, 1] | full_step | 0.961 | 0.925–0.971 |
| cpu | inverse | float32 | [2, 2] | full_step | 1.064 | 0.950–1.076 |
| cpu | inverse | float32 | [64, 64] | full_step | 1.066 | 1.052–1.099 |
| cpu | inverse | float32 | [4, 32, 32] | full_step | 1.053 | 1.048–1.183 |
| cpu | inverse | float64 | [1, 1] | full_step | 0.976 | 0.961–0.997 |
| cpu | inverse | float64 | [2, 2] | full_step | 1.057 | 1.008–1.063 |
| cpu | inverse | float64 | [64, 64] | full_step | 1.034 | 1.024–1.045 |
| cpu | inverse | float64 | [4, 32, 32] | full_step | 1.035 | 1.012–1.044 |
| node | inverse | float32 | [1, 1] | full_step | 0.938 | 0.931–0.953 |
| node | inverse | float32 | [2, 2] | full_step | 1.025 | 1.002–1.070 |
| node | inverse | float32 | [64, 64] | full_step | 1.012 | 1.001–1.018 |
| node | inverse | float32 | [4, 32, 32] | full_step | 1.023 | 1.015–1.028 |
| node | inverse | float64 | [1, 1] | full_step | 0.954 | 0.947–0.962 |
| node | inverse | float64 | [2, 2] | full_step | 1.024 | 0.982–1.039 |
| node | inverse | float64 | [64, 64] | full_step | 1.004 | 0.975–1.006 |
| node | inverse | float64 | [4, 32, 32] | full_step | 1.004 | 0.999–1.008 |
| chrome | inverse | float32 | [1, 1] | full_step | 0.932 | 0.929–0.940 |
| chrome | inverse | float32 | [2, 2] | full_step | 1.024 | 1.009–1.029 |
| chrome | inverse | float32 | [64, 64] | full_step | 1.006 | 1.004–1.012 |
| chrome | inverse | float32 | [4, 32, 32] | full_step | 1.016 | 1.014–1.021 |
| chrome | inverse | float64 | [1, 1] | full_step | 0.959 | 0.936–0.965 |
| chrome | inverse | float64 | [2, 2] | full_step | 1.015 | 1.003–1.033 |
| chrome | inverse | float64 | [64, 64] | full_step | 1.004 | 0.996–1.008 |
| chrome | inverse | float64 | [4, 32, 32] | full_step | 1.007 | 1.003–1.018 |

The prior version has no inverse JVP API. Its explicit comparator is
`P=inv(A); -(P @ E @ P)` on finite nonsingular inputs. Value and weighted
matrix/direction reverse gradients are checked, while alias/hook/layout/copy
semantics are qualified separately against native controls. JVP full steps:

| Runtime | Operation | Dtype | Shape | Mode | Candidate / comparator | Four-child range |
|---|---|---|---|---|---:|---:|
| cpu | inverse_jvp | float32 | [1, 1] | full_step | 0.963 | 0.955–0.975 |
| cpu | inverse_jvp | float32 | [2, 2] | full_step | 1.022 | 1.016–1.040 |
| cpu | inverse_jvp | float32 | [64, 64] | full_step | 1.044 | 1.033–1.056 |
| cpu | inverse_jvp | float32 | [4, 32, 32] | full_step | 1.042 | 1.019–1.049 |
| cpu | inverse_jvp | float64 | [1, 1] | full_step | 0.979 | 0.975–1.025 |
| cpu | inverse_jvp | float64 | [2, 2] | full_step | 1.033 | 1.004–1.042 |
| cpu | inverse_jvp | float64 | [64, 64] | full_step | 0.964 | 0.955–1.082 |
| cpu | inverse_jvp | float64 | [4, 32, 32] | full_step | 0.988 | 0.986–1.007 |
| node | inverse_jvp | float32 | [1, 1] | full_step | 0.895 | 0.877–0.900 |
| node | inverse_jvp | float32 | [2, 2] | full_step | 0.944 | 0.935–0.951 |
| node | inverse_jvp | float32 | [64, 64] | full_step | 1.095 | 1.079–1.107 |
| node | inverse_jvp | float32 | [4, 32, 32] | full_step | 1.068 | 1.044–1.078 |
| node | inverse_jvp | float64 | [1, 1] | full_step | 0.908 | 0.903–0.924 |
| node | inverse_jvp | float64 | [2, 2] | full_step | 0.949 | 0.937–0.962 |
| node | inverse_jvp | float64 | [64, 64] | full_step | 1.093 | 1.091–1.136 |
| node | inverse_jvp | float64 | [4, 32, 32] | full_step | 1.069 | 1.068–1.083 |
| chrome | inverse_jvp | float32 | [1, 1] | full_step | 0.888 | 0.877–0.898 |
| chrome | inverse_jvp | float32 | [2, 2] | full_step | 0.946 | 0.939–0.952 |
| chrome | inverse_jvp | float32 | [64, 64] | full_step | 1.091 | 1.088–1.095 |
| chrome | inverse_jvp | float32 | [4, 32, 32] | full_step | 1.076 | 1.067–1.082 |
| chrome | inverse_jvp | float64 | [1, 1] | full_step | 0.899 | 0.891–0.910 |
| chrome | inverse_jvp | float64 | [2, 2] | full_step | 0.962 | 0.945–0.975 |
| chrome | inverse_jvp | float64 | [64, 64] | full_step | 1.086 | 1.075–1.095 |
| chrome | inverse_jvp | float64 | [4, 32, 32] | full_step | 1.070 | 1.061–1.105 |

The largest additional matched median ratios, including forward-only losses:

| Runtime | Operation | Dtype | Shape | Mode | Candidate / comparator | Four-child range |
|---|---|---|---|---|---:|---:|
| cpu | inverse_jvp | float64 | [2, 2] | forward | 1.417 | 1.408–1.437 |
| cpu | inverse_jvp | float32 | [2, 2] | forward | 1.410 | 1.390–1.426 |
| cpu | inverse_jvp | float64 | [1, 1] | forward | 1.346 | 1.341–1.359 |
| node | inverse_jvp | float64 | [2, 2] | forward | 1.321 | 1.295–1.340 |
| node | inverse_jvp | float32 | [2, 2] | forward | 1.319 | 1.311–1.512 |
| chrome | inverse_jvp | float64 | [2, 2] | forward | 1.309 | 1.294–1.333 |
| chrome | inverse_jvp | float32 | [2, 2] | forward | 1.306 | 1.282–1.311 |
| cpu | inverse_jvp | float32 | [1, 1] | forward | 1.292 | 1.265–1.309 |
| node | inverse_jvp | float64 | [1, 1] | forward | 1.215 | 1.201–1.302 |
| chrome | inverse_jvp | float64 | [1, 1] | forward | 1.202 | 1.184–1.215 |

Complex forwards have no prior supported API and therefore have no speedup ratio:

| Runtime | Dtype | Shape | Candidate median µs |
|---|---|---|---:|
| cpu | complex64 | [2, 2] | 5.93 |
| cpu | complex64 | [32, 32] | 25.90 |
| cpu | complex128 | [2, 2] | 5.31 |
| cpu | complex128 | [32, 32] | 23.92 |
| node | complex64 | [2, 2] | 13.22 |
| node | complex64 | [32, 32] | 49.18 |
| node | complex128 | [2, 2] | 12.31 |
| node | complex128 | [32, 32] | 47.02 |
| chrome | complex64 | [2, 2] | 14.09 |
| chrome | complex64 | [32, 32] | 50.29 |
| chrome | complex128 | [2, 2] | 13.11 |
| chrome | complex128 | [32, 32] | 48.54 |

Inputs/directions/weights precede timing. Dispatch, transform, graph construction,
kernels, optional weighted backward, clearing and result destruction are timed.
No native-Torch, GPU, cold-start or universal speedup claim is made.

The four-child medians show a mixed result. WASM singleton/small inverse JVP
full steps take about 4–11% less time than the old composition. Larger/batched
WASM JVP full steps take about 7–10% more time. CPU larger float64 JVP improves
about 4%, while float32 is about 4% slower. Actual ordinary inverse CPU full
steps at sizes >=2 cost about 3–7% more; the corresponding WASM costs are about
0–3%. These native layout/guard/API corrections are adopted with their measured
costs, not described as a universal performance improvement. The largest median
regression among the 100 prior determinant/log/graph workflows is about 2.3%.
A subsequent optimization candidate is sharing the repeated upstream-gradient
matrix product between the inverse JVP reverse branches via `joint_result`;
that candidate is not implemented or measured in this version.

## Allocations, prototypes and reproduction

The 24 prior determinant/log allocation diagnostics retain separate warm and
cold-first results. Sixteen additional CPU tracemalloc rows record warm inverse
forward retained/peak bytes and a separately traced weighted-backward peak.
Inputs/directions/weights predate tracing. A later stride audit found that the
`broadcast_to(...).astype(...)` direction used in the batched allocation probe
has element strides `(1, 128, 4)`, while its C-contiguous primal has
`(1024, 32, 1)`. Native-compatible forward AD therefore copies that direction.
The 2D probe uses matching `(64, 1)` strides and needs no direction copy. This
corrects the original scope description; the archived measurements are unchanged.
The latency benchmark constructs its directions differently and is unaffected.
These are incremental tracked allocations, not total tape bytes, process RSS,
LAPACK workspace or WASM heap. New native layout and saved-value semantics differ
from the old composition, so time and allocation results must be assessed together.

| Operation | Dtype | Shape | Prior retained / peak | New retained / peak | Prior / new backward peak |
|---|---|---|---:|---:|---:|
| inverse | float32 | [64, 64] | 18,920 / 133,320 | 18,856 / 133,704 | 54,448 / 54,664 |
| inverse_jvp | float32 | [64, 64] | 56,480 / 133,320 | 55,304 / 134,880 | 89,080 / 87,536 |
| inverse | float32 | [4, 32, 32] | 18,784 / 84,016 | 18,760 / 84,400 | 54,464 / 54,720 |
| inverse_jvp | float32 | [4, 32, 32] | 56,552 / 84,016 | 71,968 / 86,368 | 89,296 / 87,488 |
| inverse | float64 | [64, 64] | 35,160 / 100,456 | 35,136 / 100,840 | 103,408 / 103,664 |
| inverse_jvp | float64 | [64, 64] | 105,616 / 139,208 | 104,456 / 135,240 | 170,936 / 169,392 |
| inverse | float64 | [4, 32, 32] | 35,168 / 51,152 | 35,144 / 51,536 | 103,552 / 103,808 |
| inverse_jvp | float64 | [4, 32, 32] | 105,704 / 139,232 | 137,504 / 168,288 | 171,152 / 169,344 |

In these warm CPU diagnostics, ordinary inverse allocations are nearly unchanged.
JVP backward peaks are about 1–2% lower, but batched JVP retained bytes increase
about 27–30%; the float64 batch forward peak increases about 21%. The retained
intermediate and corrected layout semantics have measurable costs. No universal
memory saving is claimed.

The earlier layout-copy prototype's full tests, initial root-test failure,
packages, generated assets and source snapshots remain in `prototype-a` inside
the harness. Its isolated 24-flow A/B layout pilot is development evidence only.
The final source was frozen again and every final-source gate rerun after the
copy-elimination change; the formal timing above uses those exact source hashes.
The initial adoption helper stopped safely on newly advanced affiliation docs;
the reviewed disjoint UI/docs were subsequently fast-forwarded before final
full tests. Setup-failure logs and the resumed transition are retained.

[Machine report](../../packages/torchwasm/benchmarks/results/iteration-65.json)
and [raw harness](../../packages/torchwasm/benchmarks/results/iteration-65-harness.tar.gz)
contain source/test snapshots, frozen native values/inputs, programs, failed
pilots, package artifacts, receipts and every sample. Archive: 19,073,187
bytes; SHA-256 `a88e5a6baa8307150c2980ba4460fb8d3a31f72bb1e09bbc4b1dd7b73bf50d11`.

CPU/WASM is the only backend changed this iteration. Prior WebGPU prototype and
design evidence remains separate; no new GPU performance is claimed here.
The wider independent-library/core-coverage/backend objective remains active.

## Main adoption

Validated source `914a53b67be672ee4599a6fa83e1d732c3fd6392` was adopted on remote main by ordinary non-force push.
The remote SHA matched exactly, with zero observed Actions runs for that SHA.
All 61 source/license resources, 161 active exports, 13,389 immutable payloads
and seven canonical release-policy files were rechecked. A separate proof commit
records this observation; its final main identity is checked outside the repository.
No tag, workflow dispatch, release, PyPI upload or deployment was initiated.
The broader library efficiency/core-coverage/backend objective remains active.

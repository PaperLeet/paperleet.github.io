# Torch runtime iteration 61 — stable log determinants and adjoint det solves

Version **0.61.0** adds `torch.slogdet`, `torch.linalg.slogdet`, `torch.logdet`
and their Tensor methods. Ordinary invertible real `det` backward now solves the
adjoint system instead of computing an SVD and auxiliary determinants. Source
and final integration use base main `9ab2da7b41659b95a22c76164ef253e3e8a5c495`. The full CPU/frontend/
catalog baseline is `21b3c3e7a1352486362eef0e990a2490d5716db6`; 29 affected checks at the code-reading update
and 8 supplemental checks at the inbox update, each with driver/policy/two-build
gates, qualify the later disjoint UI changes. Naming remains
StrideTorch pending the user's choice; no rename is bundled into this iteration.

## Behavior and implementation

Log determinants factor the matrix in log space, so an already overflowed or
underflowed determinant is not passed through `log`. Real float32/64 support
first-order reverse and JVP; complex64/128 support forward only. Batches, empty
matrices and strided layouts are covered. The result is a sign/logabsdet named
tuple. Output buffers, complex AD and det JVP remain explicit unsupported cases.
The [NumPy reference](https://numpy.org/doc/2.1/reference/generated/numpy.linalg.slogdet.html)
describes LU-based log determinants and their scale advantage;
[PyTorch slogdet](https://docs.pytorch.org/docs/2.14/generated/torch.linalg.slogdet.html)
and [logdet](https://docs.pytorch.org/docs/2.14/generated/torch.logdet.html) define
the public entry points. The concrete derivative and mutation behavior here is
verified against native PyTorch **2.14.1**, git
`5c4886908584029761b579af026dcfb627c84070`, rather than assumed from newer main.

Real slogdet sign is a differentiable output with an unused edge. Backward on
sign alone leaves input `.grad` as `None`, while both outputs share saved-graph
lifetime. The graph engine now supports an unused VJP edge without converting
it into a zero array, and releases its saved ancestors consistently with the
tested grad/backward frontiers. Input/sign saved-value mutation guards,
retained graphs and storage release are tested. Logabs output edits follow the
native unsaved-output behavior.

Real negative logdet gives NaN with zero slope. Singular log derivatives are
undefined; pivoted IEEE fallback solves preserve the tested Inf/NaN outcomes
without clipping or replacing zero pivots. JVP uses an immutable factor input
snapshot. Reverse through the JVP differentiates the direction and has no edge
to the original matrix, matching the pinned native saved-factor implementation.
This is not a claim of arbitrary higher derivatives or future-version parity.
The [pinned derivative source](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/csrc/autograd/FunctionsManual.cpp)
is the implementation reference.

NaN pivot propagation is probed once on import with a 2×2 matrix. NumPy 1.26
CPU and the tested WASM build can otherwise misclassify NaN as a zero pivot.
Those runtimes scan for NaN batches; newer CPU NumPy avoids that scan. Complex
logdet assigns real/imaginary components separately: singular real `-Inf` must
survive a NaN phase. Qualification checks the two components independently;
complex `equal_nan` alone previously concealed a wrong real component.

Existing singular/extreme det polynomial-cofactor gradients and finite-real-input
restriction are retained and qualified separately from native log derivatives.
Raw log VJP/JVP reverse results match native column-major matrix layout. Stored
leaf gradients retain owned buffers. Native can reuse different positive
singleton-axis strides; ten arrays use the documented equivalent layout contract,
with all meaningful non-singleton strides checked exactly.

## Correctness and integration evidence

- Final frozen-source CPU regression: **40,970 passed, 1 skipped**. Source hashes
  are captured before and checked after the test process.
- Actual Node/Pyodide and Chrome Worker: **7,340 contracts each**, 54 independent
  suites, Pyodide 0.29.4 / NumPy 2.2.5. Chrome 155.0.8059.82.
- Installed public wheel on Python 3.10.19 / NumPy 1.26.4: **7,353 passed**,
  with native Torch absent. Public wheel/sdist and private vendored wheel match
  all **59 Python modules + LICENSE/NOTICE** byte for byte.
- Public/private packages coexist, each survives uninstalling the other, and
  aliases/state remain isolated. New logs, sign-only None gradients, JVP reverse
  and detached namedtuple pickle are checked in both installed namespaces.
  Fresh-process activation/model restores and native Torch isolation also pass.
  Direct pickle of live autograd graph outputs retains the existing limitation;
  the failed development probe is preserved, not reported as a passing check.
- **600 frozen cases per CPU/Node/Chrome**, also repeated on the isolated
  Python 3.10 / NumPy 1.26.4 installed wheel, from 200 exact input/seed/direction
  arrays: 480 native log cases; 54 ordinary native det cases; 48 separate legacy
  cofactor cases; 18 retained input-restriction cases. Each runtime checks 1,764
  numeric arrays and 360 None-gradient fields. Real logs include VJP, JVP and
  reverse through direction; complex cases are forward only. C/F/sliced layouts,
  batches, empty shapes, singular/negative determinants, tiny/huge scales and
  NaN/Inf are represented. The 48/18 legacy cases are **not native parity passes**.
- All **78 exported modules and 342 mutants** pass under original browser budgets.
  The 1,152 frozen native catalog input hashes are unchanged. All 54 independent
  suites are embedded in and checked by the exported application runtime.
- Full frontend checks, release-policy checks and both build profiles pass at
  the recorded full-validation baseline. The affected UI checks and driver/
  policy/two-build gates also pass at the final main UI base. After final runtime refresh, the WASM driver and
  both build profiles are repeated; numerical catalog tests use that final export.
  All **12,599 previously published payloads** remain byte-identical. The 158
  new assets are reachable; 161 active export/index identities are recorded.
- Seven release-policy files remain byte-identical to canonical
  `13d2782a494415ba72bc604a0ef7e4c4f875ba25`. No CI trigger, tag, dispatch,
  release, package upload or deployment is initiated.

Development failures and pilots remain in the raw archive. An owned pre-final
CPU run was explicitly stopped after its capability-probe source became obsolete;
its termination is recorded separately from the final completed regression.

## Performance against 0.60

Twelve fresh children (four each CPU/Node/Chrome) run 100 workflows each:
**88 matched** ordinary SPD determinant/log-composition or graph controls and
**12 candidate-only** complex forward workflows. Old logs are `det.log` and
`det.abs.log` compositions on positive SPD matrices; they are not equivalent
negative/singular APIs. Each child uses common calibrated batches of at least
20 ms and five alternating samples. Tables show the median of four per-child
ratios; below 1 is faster. Inputs/conversions/seeds are outside timing; dispatch,
graph construction, kernels, weighted backward and gradient clearing are inside.
Heavy correctness gates finish before formal timing. An observed unrelated local
WASM catalog job is also allowed to finish before timing; host-wide background
activity is not controlled. CPU thread caps are two.
There is no native-Torch, GPU, cold-start or universal-throughput comparison.

64×64 determinant forward + weighted backward:

| Runtime | Operation | Dtype | Shape | Mode | Candidate / baseline | Four-child range |
|---|---|---|---|---|---:|---:|
| cpu | det | float32 | [64, 64] | full_step | 0.198 | 0.197–0.198 |
| cpu | det | float64 | [64, 64] | full_step | 0.275 | 0.272–0.290 |
| node | det | float32 | [64, 64] | full_step | 0.235 | 0.232–0.266 |
| node | det | float64 | [64, 64] | full_step | 0.326 | 0.324–0.328 |
| chrome | det | float32 | [64, 64] | full_step | 0.236 | 0.236–0.238 |
| chrome | det | float64 | [64, 64] | full_step | 0.332 | 0.332–0.334 |

Small determinant forward control:

| Runtime | Operation | Dtype | Shape | Mode | Candidate / baseline | Four-child range |
|---|---|---|---|---|---:|---:|
| cpu | det | float32 | [2, 2] | forward | 0.993 | 0.990–1.000 |
| cpu | det | float64 | [2, 2] | forward | 0.986 | 0.973–1.006 |
| node | det | float32 | [2, 2] | forward | 1.007 | 0.979–1.017 |
| node | det | float64 | [2, 2] | forward | 1.015 | 1.007–1.016 |
| chrome | det | float32 | [2, 2] | forward | 1.008 | 1.006–1.022 |
| chrome | det | float64 | [2, 2] | forward | 1.007 | 0.985–1.012 |

Unchanged scalar/vector graph controls:

| Runtime | Operation | Dtype | Shape | Mode | Candidate / baseline | Four-child range |
|---|---|---|---|---|---:|---:|
| cpu | graph | float32 | [1] | full_step | 1.010 | 0.985–1.036 |
| cpu | graph | float32 | [32] | full_step | 1.021 | 1.001–1.039 |
| cpu | graph | float64 | [1] | full_step | 1.012 | 1.002–1.053 |
| cpu | graph | float64 | [32] | full_step | 0.999 | 0.990–1.004 |
| node | graph | float32 | [1] | full_step | 1.007 | 0.972–1.009 |
| node | graph | float32 | [32] | full_step | 0.999 | 0.996–1.007 |
| node | graph | float64 | [1] | full_step | 1.004 | 0.995–1.015 |
| node | graph | float64 | [32] | full_step | 0.999 | 0.994–1.003 |
| chrome | graph | float32 | [1] | full_step | 1.002 | 1.000–1.004 |
| chrome | graph | float32 | [32] | full_step | 1.013 | 1.000–1.034 |
| chrome | graph | float64 | [1] | full_step | 1.000 | 0.984–1.009 |
| chrome | graph | float64 | [32] | full_step | 1.004 | 0.996–1.017 |

The 64×64 logdet forward-only comparisons cost about 1.10–1.18 times the old
composition. Small forward-only logdet costs reach about 1.72 times; these
new-API common-path costs are a follow-up optimization target. Existing small
det forward is approximately unchanged, while full determinant/log backward
workflows benefit from the solve path. Graph-control medians include small
losses and remain visible below.

Largest matched ratios, including any losses:

| Runtime | Operation | Dtype | Shape | Mode | Candidate / baseline | Four-child range |
|---|---|---|---|---|---:|---:|
| chrome | logdet | float32 | [1, 1] | forward | 1.723 | 1.714–1.744 |
| chrome | logdet | float64 | [1, 1] | forward | 1.712 | 1.703–1.732 |
| chrome | logdet | float64 | [2, 2] | forward | 1.706 | 1.673–1.735 |
| chrome | logdet | float32 | [2, 2] | forward | 1.701 | 1.664–1.724 |
| node | logdet | float32 | [1, 1] | forward | 1.694 | 1.671–1.739 |
| chrome | logdet | float32 | [8, 8] | forward | 1.693 | 1.686–1.706 |
| chrome | logdet | float64 | [8, 8] | forward | 1.677 | 1.664–1.696 |
| node | logdet | float64 | [2, 2] | forward | 1.674 | 1.663–1.685 |

All 300 runtime/workflow aggregate rows, child samples, candidate-only timings,
source hashes and calibration values are retained in the machine report/archive.
Float32 old/new SVD-versus-solve gradients have a separately labeled 1e-5
normwise comparison. Native accuracy uses the independent frozen oracle above.

## Memory and footprint

Warm CPU tracemalloc incremental tracked bytes, with inputs/seeds allocated before
tracing. Both source namespaces and all measured forward/backward paths are
primed by the archived `warm-memory.py` driver. The first-cold diagnostic is
retained separately; NumPy lazy initialization is not counted as tape savings. Forward retained/peak and backward peak are separate measurements;
these are not total tape bytes, RSS, LAPACK workspace or WASM heap size.

| Variant | Operation | Dtype | Shape | Newly retained after forward | Forward peak | Backward peak |
|---|---|---|---|---:|---:|---:|
| baseline | det | float32 | [64, 64] | 2,683 | 67,624 | 309,652 |
| baseline | det | float64 | [64, 64] | 2,439 | 34,192 | 276,584 |
| candidate | det | float32 | [64, 64] | 2,755 | 67,664 | 187,620 |
| candidate | det | float64 | [64, 64] | 2,223 | 34,232 | 138,380 |

The full 24-row allocation table is in the machine report. Source/license bytes
change from 722,071 to 731,240; local
concatenated gzip from 166,943 to
169,434 bytes. Gzip is a local footprint diagnostic,
not wheel or network transfer size.

## Reproduction and limits

[Machine report](../../packages/stridetorch/benchmarks/results/iteration-61.json)
and [raw harness](../../packages/stridetorch/benchmarks/results/iteration-61-harness.tar.gz).
Raw archive: 7,438,519 bytes, SHA-256 `18fe6c6a118fa6ff5b4c61bc6aee2164524e56387387ad416b2fe3ded8bd7e4e`. It contains baseline
and final sources, frozen native arrays/results, programs, runtime plans,
package artifacts, child receipts, full outputs and preserved development trials.
The repository benchmark is
`packages/stridetorch/benchmarks/bench_determinants.py`.

This iteration remains dense CPU/WASM. No WebGPU speedup or backend parity is
claimed. A diagnostic also reproduces an existing NumPy 2.5.4 strict-error-policy issue:
64×64 identity det raises a floating-point exception under `numpy.seterr(all="raise")`
in both baseline and candidate. It is not introduced by this change and remains
a follow-up; default-policy warnings and both probes are retained.
Existing half/complex AD, unsupported API and prior numerical limits
remain in force. GPU routing and independent-library quality remain part of the
active broader goal. This local milestone does not mark that goal complete.

## Main adoption

Validated source `9526e0379f576df49ed6e0a1e52aabc666b47d9c` was adopted on remote main by ordinary non-force push.
The remote SHA matched exactly, with zero observed Actions runs for that SHA.
All 61 source/license resources, 161 active exports, 12,757 immutable payloads
and seven canonical release-policy files were rechecked. A separate proof commit
records this observation; its final main identity is checked outside the repository.
No tag, workflow dispatch, release, PyPI upload or deployment was initiated.
The broader library efficiency/core-coverage/backend objective remains active.

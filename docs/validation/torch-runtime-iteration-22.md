# StrideTorch 0.22: shared real statistics

This version adds shared real `var/std/var_mean/std_mean` kernels, strict
reduction/overload contracts and analytical VJP/JVP. Main includes 0.22 at
`7eac78ab`; its exact Quality run passes native, complete WASM and pilot checks.
All local release gates also pass. Runtime-identical curriculum descendant
`db6408fd` passes its exact Quality and eligible Pages deployment; all 78 live
modules and 158 immutable asset comparisons pass.

## Correctness and computation

Dimension parsing rejects noninteger, out-of-range and repeated axes, while
supporting scalar and empty tensors, empty dimension lists, fractional correction,
`keepdim` and the legacy `unbiased` overload. Forward variances clamp the degrees
of freedom to zero. Reverse rules preserve the independently tested distinction
between omitted and explicit empty dimensions. Negative/NaN/infinite corrections,
nonfinite inputs, constant standard deviations and mixed storage/dtypes after
model conversion have native-reference coverage. Complex statistics and output
buffers retain explicit unsupported errors.

The [PyTorch variance definition](https://docs.pytorch.org/docs/2.14/generated/torch.var.html)
and [native reverse rules](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/csrc/autograd/FunctionsManual.cpp)
help identify these contracts; executable comparisons use native PyTorch 2.14.1.
Paired float32 moments accumulate in double precision. A Welford fallback handles
exceptional paired rows; ordinary rows retain vectorized two-pass computation.
Backend reduction order is not promised bitwise identical across platforms.

Variance saves the input version, not an unsaved variance output. Standard
deviation saves both input and result; paired mean pullbacks retain the same
necessary guards. Mutating a detached mean output does not corrupt the variance
pullback. Saved means are independent, output-sized copies; centered input-sized
arrays are not retained by these pullbacks. Weak references detect source storage
rebinding and trigger recomputation. No gradients/seeds are cached across calls.

One forward numerical scope and the existing reverse-engine numerical scope
replace redundant nested NumPy error-state contexts. User hook/error-policy
behavior remains checked. JVP supports float32/float64 and reverse differentiation
of its tangent, without enabling nested JVP or general higher-order backward.

## Validation so far

The original 1,584-case audit had 796 differences, including 480 exception-type,
108 shape, 276 value and 78 gradient differences (categories overlap). The same
finite boundary matrix passes after the implementation. A pre-cache complete
infrastructure run passed 25,389 tests with one optional Docker skip and 17
warnings (188.04s); this is not reused as the final source's complete result.

After merging main at `ac360bd8`, the final source passes 25,605 infrastructure
tests with one optional Docker skip and 17 warnings (202.29s). All 16 upstream
UI/catalog files remain byte-identical; numerical source hashes still match the
timed/runtime snapshots. Fresh private and standalone wheels/source archives
contain identical core/license bytes and have disjoint installed paths; the
private distribution retains current curriculum metadata.

The original native catalog also passes 78 problems, 10,218 test executions over
three repetitions and 342 mutants (135.72s), retaining its existing per-module
limits and 600s aggregate budget. All 78 exported WASM modules pass with no
browser budget violations (540.251s aggregate). Flow Matching takes 128.446s
calibration, 40.101s learner execution and 171.204s total, within its unchanged
240/60/300s limits. These application times are correctness/budget evidence,
not a controlled comparison with earlier releases.

Export preserves all 6,282 published immutable assets byte-for-byte. All 158
new immutable assets are reachable; the release freezes 161 files including
the three catalog indexes. Both runtime profiles carry Apache-2.0/NOTICE.
The frontend check, production build and Pages policy check pass, including a
repeat after preserving the latest upstream handbook UI at `2ebe451f`. Its four
changed files match upstream exactly; numerical, packaging and exported runtime
inputs remain identical to the validated snapshots. The repeated production
build takes 5.38s; its initial JavaScript is 407,693 raw / 131,367 gzip bytes.

The current source also passes 4,031 focused native/statistics/model-conversion/IEEE
and standalone contracts. Its actual Node/Pyodide and Chrome Worker runs each
pass 767 independent contracts, and the independently installed wheel passes
787 tests on Python 3.10.19/NumPy 1.26.4 without native Torch. All 41 Python
modules and two license resources match their frozen runtime/wheel/sdist hashes.
Independent checks include closed-form moments, finite directional derivatives,
VJP/JVP duality, repeated backward/hooks and train/checkpoint/resume behavior.

Retained failures include the initial overload/fixture mistakes, NaN-correction
outputs, one post-forward float64-to-float32 underflow regression, 36 nonfinite
paired-output mismatches and three subsequent float64 reverse-cache mismatches.
A Chrome bootstrap failed fetching the pinned exceptiongroup package before
any tests ran. The local file matched its lock SHA and served identical bytes;
a fresh labeled Worker run passed all 767 tests. Both records are retained.

## Controlled comparison with 0.21

Each of three phases freezes source, resources and identical measured harnesses,
then runs 16 fresh processes: CPU/WASM, float32/float64, baseline/candidate/
candidate/baseline. Thirteen fixed workloads retain eleven timing batches after
three warmups; there are 400 small forward calls, 100 row-reduction calls or 40
reverse calls per batch. Source and harness hashes are checked around each child.
All 48 children finish successfully. Tests/builds/audits do not overlap these
local comparative timings. Thread limits request two; actual BLAS threading is
not established on this shared machine.

The baseline paired workload composes its existing variance/std and mean calls;
the candidate shares moments. Fixtures use matching valid values/gradients and
exclude corrected scalar/empty/degenerate behavior. This compares steady-state
library workloads, not native Torch, application speed, cold start or peak memory.

The initial phase improves variance but slows standard-deviation reverse by
6–13%. Output-sized mean caching retains 9–15% costs. The final phase removes
redundant error-state contexts; all earlier samples and regressions remain.
Final relative time changes across float32/float64 are:

| Workload | CPU | WASM |
| --- | --- | --- |
| Small variance | 54–55% lower | 55–56% lower |
| Row variance | 42–46% lower | 47–50% lower |
| Variance reverse | 42–43% lower | 42–44% lower |
| Standard-deviation forward | 21–25% lower | 18–25% lower |
| Paired variance forward | 33–42% lower | 37–44% lower |
| Paired variance reverse | 26–28% lower | about 28% lower |
| Standard-deviation reverse | 2–6% higher | 5–6% higher |
| Paired standard-deviation reverse | 2–6% higher | 4–7% higher |

The unchanged ordinary reverse control is within 1% on CPU and 2–3% slower on
WASM. Shared-machine measurements do not establish a causal application effect.
The remaining standard-deviation reverse cost is explicit, not counted as a win.
Maintained CPU and WASM entry points both execute all 13 statistics cases.

[Evidence](../../packages/torchwasm/benchmarks/results/iteration-22.json) and
[source/harness archive](../../packages/torchwasm/benchmarks/results/iteration-22-harness.tar.gz)
retain all phases, numerical inputs, rejected failures and current source.
The [final local gates](../../packages/torchwasm/benchmarks/results/iteration-22-local-gates.json)
and [release snapshot archive](../../packages/torchwasm/benchmarks/results/iteration-22-local-gates-harness.tar.gz)
retain full native/WASM results, packaging receipts, history checks and frozen
runtime/live-verification inputs. The independent project name **StrideTorch**
is explicitly confirmed by the human; distribution, import and launcher use
`stridetorch`, while host/launcher aliases preserve ordinary `import torch`.

## Verified main quality and publication

[Exact Quality 38879271337](https://github.com/PaperLeet/paperleet.github.io/actions/runs/38879271337)
passes all three required jobs. Linux infrastructure passes 25,605 tests with
one optional skip and 20 warnings (564.42s). Native catalog passes 78 problems,
10,218 executions, 342 mutants and three repetitions (419.80s). Actual WASM
passes all 78 modules without browser budget violations (1,025.912s aggregate).
Flow Matching takes 233.066s calibration, 46.635s learner and 284.134s total,
within unchanged 240/60/300s limits. These timings are budget observations on
the actual runner, not controlled performance comparisons.

Main advances to `db6408fd` before the original Quality finishes, so the old
Pages trigger is skipped/superseded rather than published. All 43 core/license
hashes, library packaging inputs, problems, tests, workflows and browser bundles
are identical in that descendant. Nine upstream curriculum/UI files differ,
including private application catalog metadata; whole-application wheel identity
is not claimed. Its
[Quality 38880784326](https://github.com/PaperLeet/paperleet.github.io/actions/runs/38880784326)
and eligible
[Pages 38883159210](https://github.com/PaperLeet/paperleet.github.io/actions/runs/38883159210)
succeed, including every actual build/deploy step. The descendant WASM artifact
also passes all 78 modules without budget violations (1,007.261s aggregate).

Live verification decodes compressed starters and checks all 78 starters,
manifests and module details, then compares all 158 immutable payloads exactly.
Both runtime profiles carry the expected 0.22 version and Apache-2.0/NOTICE.
The original skipped Pages observation remains separate from the actual eligible
deployment. [Main-quality evidence](../../packages/torchwasm/benchmarks/results/iteration-22-main-quality.json),
[remote adoption](../../packages/torchwasm/benchmarks/results/iteration-22-remote-adoption.json)
and [remote/live archive](../../packages/torchwasm/benchmarks/results/iteration-22-remote-adoption-harness.tar.gz)
retain both workflow histories and frozen verification inputs. This proves
repository/site adoption; no public PyPI publication has occurred.

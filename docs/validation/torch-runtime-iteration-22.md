# StrideTorch 0.22 candidate: shared real statistics

This development candidate adds shared real `var/std/var_mean/std_mean` kernels,
strict reduction/overload contracts and analytical VJP/JVP. It is not adopted on
main. Final infrastructure, full application, packaging/export and integration
gates remain pending. Version 0.21 is already deployed and independently verified.

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

The current source passes 4,031 focused native/statistics/model-conversion/IEEE
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

[Evidence](../../packages/stridetorch/benchmarks/results/iteration-22.json) and
[source/harness archive](../../packages/stridetorch/benchmarks/results/iteration-22-harness.tar.gz)
retain all phases, numerical inputs, rejected failures and current source.

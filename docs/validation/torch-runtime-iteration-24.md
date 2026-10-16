# StrideTorch 0.24 candidate: typed means and half precision

This candidate enables `torch.mean(dtype=...)` and repairs existing half-precision
means. Main still uses 0.23. The candidate has not been integrated or deployed;
complete exported WASM and latest UI integration have passed. Main and remote
adoption remain release gates.

## Behavior and computation

Both method and functional means accept explicit floating/complex output dtype,
including floating means of integer/boolean input. Strict dimension parsing
retains integer/tuple/list axes, scalar and empty inputs, and empty-axis global
reductions. Boolean/fractional axes, duplicate/range errors and invalid keepdim
overloads follow the tested native contracts. An explicit keepdim requires an
explicit dimension, including None. Shape checks remain per-call under the
existing bounded public-dispatch cache.

The former implementation returns infinity for the half mean of 65,536 ones,
and can erase its subnormal reverse gradient by overflowing the denominator.
Half outputs now sum/divide in float32 and round the output once. Half reverse
seeds divide in float32, round into their output dtype, then follow the existing
input-gradient cast. Wider input is not first rounded into half. This follows
[native CPU mean](https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/native/ReduceOps.cpp#L1315-L1362),
and is independently compared against installed native PyTorch 2.14.1.

The final half path uses a precomputed count and output-sized sum/division;
the library does not explicitly materialize a full-input float32 copy. NumPy's
internal buffering is not a measured peak-memory guarantee. Pullbacks save shape/count,
without retaining intermediate input values or guarding unsaved output values.
Leaf ownership still retains storage needed for gradient accumulation. Detached
input/output mutation and repeated backward retain native mean behavior.
Post-forward model conversion preserves native seed rounding and gradient dtype.

Float32/float64 JVP includes casts between those dtypes and reverse gradients
through the tangent. Half/complex JVP, complex autograd and output buffers remain
explicit unsupported boundaries. Existing complex forward casts are checked.
Numerical contexts restore caller error policy; WASM trap assertions match the
pinned NumPy runtime's actual FPU behavior rather than assuming native traps.

## Correctness evidence

A 2,018-case native boundary matrix starts with 837 differing cases: 763
exception differences, 74 value differences and 50 gradient differences;
categories overlap. The initial audit's half-gradient absolute tolerance masked
subnormal errors. Both earlier records remain, while the final audit uses zero
half absolute tolerance for values and gradients. It reports no differences.

The final full infrastructure suite passes 25,928 tests, with one optional Docker
skip and 23 warnings (185.54s). The pre-optimization source separately passes the
same count (186.58s); that earlier result is not substituted for the final source.
Focused native and independent mean contracts pass 315 tests, including mixed
casts, noncontiguous inputs, integer/boolean means, half overflow, precise
subnormal pullbacks, JVP/reverse-over-JVP and model conversion.

Final Node/Pyodide 0.29.4/NumPy 2.2.5 and Chrome 155 Worker runs each pass 784
independent contracts. The installed wheel passes 804 tests on Python 3.10.19/
NumPy 1.26.4 without native Torch. All 41 Python modules and two license resources
match standalone/private wheel/source-archive bytes with disjoint installed paths.
A subsequent documented wheel changes only its METADATA description and RECORD:
all runtime resources, entry points and metadata headers are byte-identical to
the wheel tested with 804 contracts. README is verified in the new description
and source archive. That documentation-only artifact is not presented as a
separately repeated 804-test run.

Independent checks use analytical means/gradients, exact half subnormal results,
directional tangent identities, output mutation and intermediate-array lifetime.
They do not derive every expected result from another adapter operation.

## Retained failures and rejected work

Initial focused contracts have five failures: four expose an incorrect initial
half cast-before-accumulation assumption, and one attempts unsupported direct
non-leaf mutation. Native executable comparisons and the CPU source identify
the correct opmath route; the mutation fixture uses a detached alias.
A subsequent lifetime fixture wrongly expects the graph's leaf to be collected;
it is corrected to check reclaiming an intermediate array while retaining its
leaf for accumulation. Initial source, tests and failure logs remain preserved.

An explicit float32 input-copy trial passes all 315 contracts but adds 16–18%
WASM half-row latency relative to the first correct kernel, with extra input-sized
storage. It remains an external rejected source and is never adopted. A no-copy
sum/divide trial passes the same contracts and lowers half-row time by about
9–10% on WASM and 18% on CPU relative to that first correct kernel. The adopted
source then receives its own complete numerical/distribution gates.

## Controlled comparison

Four phases retain 64 successful fresh sequential children: CPU/WASM,
float32/float64 and baseline/candidate/candidate/baseline ordering. Source and
harness hashes are checked around every child. No local tests, builds or audits
overlap timing producers. The first/final phases compare frozen 0.23 with 0.24;
two middle phases compare the first correct 0.24 kernel with external trials.

All twelve fixed cases use eleven batches after three warmups: 1,000 small calls,
250 row/column calls or 100 reverse calls per batch. NumPy values and analytical
gradients are checked before timing. Baseline method dtype already works;
new functional dtype and corrected overflow/invalid cases are excluded from
speed comparisons. CPU environment controls request two threads without proving
actual BLAS threading on the shared host. These are steady-state library calls,
not native Torch, cold-start, peak-memory or application speed measurements.

Final relative time observations against 0.23 are:

| Fixed case | CPU | WASM |
| --- | --- | --- |
| Row/column means | 7–12% lower | 6–10% lower |
| Valid half rows | 33–34% lower | within about 0.5% |
| Small method/function means | 2–6% lower | 3% higher to 3% lower |
| Mean reverse workloads | 5–7% lower | 2–8% lower |

The unchanged exponential control is 3–5% faster on CPU and ranges from about
2% slower to unchanged on WASM. Small changes within that drift are not strong
causal gains. The float64 WASM small method case is 2.9% slower, and remains
explicit rather than counted as a win. Earlier half-row and copy-trial costs
are retained alongside the final samples. No universal speedup is claimed.

The [development evidence](../../packages/stridetorch/benchmarks/results/iteration-24.json)
and [source/test/harness archive](../../packages/stridetorch/benchmarks/results/iteration-24-harness.tar.gz)
retain all four phases, failed fixtures/prototypes, final source and distribution
receipts. They explicitly leave original application/export/remote gates pending.

## Application integration before full exported WASM

The original native catalog passes 78 problems, 10,218 executions, 342 mutants
and three repetitions in 126.96s, retaining its existing T3 limits and 600s
aggregate budget. Export preserves all 6,598 published immutable assets; every
one of the 158 new assets is reachable. The 161-file release snapshot is frozen.
Autograd runtime is 462,797 raw bytes (107,060 with local gzip estimation);
schedules is 37,343 raw bytes. Both include Apache-2.0/NOTICE.

Upstream at `d968be6e` changes nine UI/documentation files; all are preserved
byte-for-byte and all 43 core/license hashes remain identical to the numerical
and timing evidence. Fresh private distributions retain current catalog/server
and identical standalone core bytes. Complete frontend checks validate 48
bilingual guides/144 prompts; production build (5.44s) and Pages policy pass.
Initial JavaScript is 408,091 raw / 131,542 gzip bytes.

The [integration gates](../../packages/stridetorch/benchmarks/results/iteration-24-integration-gates.json)
and [integration snapshot archive](../../packages/stridetorch/benchmarks/results/iteration-24-integration-gates-harness.tar.gz)
retain those receipts. Complete exported WASM now passes all 78 modules with no
browser budget violations (355.605s aggregate). Flow Matching takes 84.719s
calibration, 17.001s learner and 103.320s total, within unchanged 240/60/300s
limits. These application times are budget observations, not controlled
performance comparisons with another release.

The [complete local gates](../../packages/stridetorch/benchmarks/results/iteration-24-local-gates.json)
and [final WASM receipts](../../packages/stridetorch/benchmarks/results/iteration-24-local-gates-harness.tar.gz)
close the catalog gate against frozen source and all 161 export hashes. The later
eight-file UI change at `1be5db7e` is preserved byte-for-byte. Repeated complete
frontend checks, production build (8.84s) and Pages policy pass; initial JavaScript
is 408,091 raw / 131,549 gzip bytes. Core/license, Python packaging, tests,
problems, workflows and all 161 export hashes remain identical, preserving the
completed numerical/catalog evidence without repeating unrelated workloads.

[Latest upstream gates](../../packages/stridetorch/benchmarks/results/iteration-24-latest-upstream-gates.json)
and the [affected-check archive](../../packages/stridetorch/benchmarks/results/iteration-24-latest-upstream-gates-harness.tar.gz)
close that integration gate. Main adoption and exact remote Quality/eligible
Pages/live verification remain pending.

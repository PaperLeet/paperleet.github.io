# StrideTorch 0.24 candidate: typed means and half precision

This candidate enables `torch.mean(dtype=...)` and repairs existing half-precision
means. Main still uses 0.23. The candidate has not been integrated or deployed;
original catalogs, exported WASM and remote adoption remain release gates.

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
there is no full-input float32 conversion buffer. Pullbacks save shape/count,
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

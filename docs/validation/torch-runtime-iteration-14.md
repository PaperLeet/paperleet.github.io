# Lightweight torch 0.14 — unreleased work in progress

The maintained implementation is being expanded to cover ordinary arithmetic
function/method APIs, mixed dtype/Number promotion and checked alpha values.
**This candidate is not adopted on main.** Numerical edge cases now have
corrections and independent regression tests; final distribution, browser and
performance gates remain outstanding.

## Implementation and evidence so far

`torch.add/sub/mul/div`, `subtract/multiply/divide/true_divide` and corresponding
Tensor methods are implemented. Add/sub accept alpha, division accepts floor or
trunc rounding, and actual output buffers remain explicitly unsupported. Integer
true division uses default float32; rounded int64 division avoids passing through
floating storage. Tensor/Number rank/category priority shares the 0.13 promotion
rules. Arithmetic scalar operands wrap integer storage and allow floating overflow,
while alpha/where storage keeps separately checked conversion rules.

Operator and functional reflected division use separate promotion paths, following
native overload behavior. Boolean subtraction is checked before JVP constant
coercion. Complex storage preserves components and native non-finite add behavior;
ComplexHalf and unvalidated complex reverse graphs fail explicitly. Rounded VJPs
save only shape tuples and return zero gradients, without retaining operand arrays.

The [native binary metadata rules](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/BinaryOps.cpp)
and [TensorIterator](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/TensorIterator.cpp)
guide implementation. Independent Torch 2.14.1 matrices compare dtype, value,
errors, methods/functions/operators, Number protocols, alpha, rounded division,
VJP, JVP and reverse-over-forward behavior. NumPy-only workflows separately check
analytic gradients, two-layer SGD updates, storage release and aliases.

The initial 35-file phase passes **16,743 tests**, one optional Docker skip and
seven warnings in 173.74 seconds; minimum NumPy 1.26.4 passes 144 standalone tests,
and real Pyodide 0.29.4/NumPy 2.2.5 passes 124 contracts. This phase is preserved
and reconstructed exactly; it precedes the graph-boundary changes and optimizations.
Current optimized source passes **12,319 focused tests**, **145 minimum-NumPy
standalone tests** without native Torch and **125 pinned WASM contracts**. These
focused checks do not replace the required final full application regression.

## Exploratory CPU performance

Baseline is released 0.13 commit `594a984d8fed2fbcba933e946e8177922c742871`,
with 34 hashed Python files. Initial and optimized candidates each have 35 hashed
files. Both notices and all benchmark harnesses are hashed. All owned correctness/
build processes are terminal before sequential timings; no dependent source or
harness edits occur during timing.

The first two protocols retain every sample independently: baseline→initial→initial→baseline,
then baseline→initial→optimized→optimized→initial→baseline, for float32 and float64.
There are 16 and 24 fresh CPU processes respectively, with 13 arithmetic cases and
seven existing runtime controls. CPU uses Python 3.12.11/NumPy 2.5.4/macOS arm64 /
Apple Accelerate. BLAS limits precede imports and the single-thread policy is
queried before/after. Eleven batches follow three warmups, with 1,200 calls for
forward cases, 120 for single backward/control cases and 30 for the 16-stage chain.
Fixture construction and independent value/gradient checks are outside timing;
operation/allocation/cleanup is included. These are exploratory CPU data, not
final browser performance or a general speedup claim.

Initial scalar calls cost approximately 26–30% more than 0.13 in the first protocol;
the affine chain costs 14–17% more. Immutable Number operands now keep private
array values directly, avoiding a temporary Tensor and storage-version allocation.
No-grad paths skip unnecessary derivative closures, and exact builtin Number
parsing has a direct path. User Tensor operands keep their mutation guards.

In the three-variant protocol, optimized scalar add is +1.0%/−0.3%, multiply
+4.4%/+6.1%, division +2.9%/+3.5% and reflected division +4.3%/+4.9%
(float32/float64, relative to 0.13). Reflected subtraction is about 40–43% faster
because it no longer builds a negate/add pair. Ordinary Tensor/Tensor calls still
cost about 6–19% more, and the affine backward chain +6.0%/+4.6%. Attention, MLP,
AdamW and slice/index controls stay within about 2%. Sub-microsecond no-op ratios
are not interpreted. Original process variation and all initial regressions are
retained; no series is filtered or pooled to improve the result.

A third balanced protocol compares baseline→optimized→refined→refined→optimized→
baseline with the same frozen harnesses (24 more fresh CPU processes). Refined
same-dtype Tensor operands bypass redundant array conversion calls. Checked
integer wrapping and ordinary float32/float64 scalar storage avoid error contexts
when conversion cannot signal; extreme/subnormal conversions keep isolated error
handling. All source and harness hashes are verified again.

Against 0.13 in this third series, scalar addition changes −10.3%/−9.9%, scalar
multiplication −9.9%/−4.9%, scalar division −9.0%/+0.3% and reflected subtraction
−47.3%/−45.4% (float32/float64). Tensor/Tensor arithmetic stays within roughly
0–6% of baseline; backward calls within roughly 3%. The affine backward chain
changes −2.8%/−2.5%, and runtime controls remain near parity. These observations
apply to the fixed ordinary-value fixtures. In particular the reflected-division
implementation still needs the IEEE correction below, which may change costs.
There are now 64 fresh CPU process reports across three separate protocols;
all earlier samples and regressions are retained. No final WASM/Chrome speedup
or resolved cause of process variation is claimed.

Full samples, frozen source/resource/harness hashes, source/test material and
verified reconstruction patches are in
[iteration-14-prototype.json](../../packages/paperleet-torch/benchmarks/results/iteration-14-prototype.json).
Local frozen artifacts and logs remain in
`paperleet-wasm-evaluation-20261015/torch-iteration-14`.

## IEEE and callback corrections under validation

Native `Tensor.__rdiv__` composes reciprocal and multiply. The direct division
previously used for Python reflected division failed at subnormal denominators:
`1e-300 / tensor([1e-310], dtype=float64)` must overflow, although functional
`torch.div(1e-300, tensor)` remains approximately `1e10`. The corrected operator
preserves reciprocal storage rounding, multiplication order and reciprocal's
square-first derivative. `torch.reciprocal` and the Tensor method are also
available. Complex zero sign handling follows native results, while complex
reverse graphs remain explicitly unsupported.

Forward AD omits absent constant tangents instead of fabricating `0 * infinity`.
Explicit dual zero tangents still take part in the calculation, with native
non-finite and signed-zero behavior. Constant broadcast/mixed-dtype directions
match the primal result's shape and dtype.

An initial correct implementation scoped NumPy errors around individual reverse
nodes. Full application regression passed **16,887 tests**, one optional Docker
skip and seven warnings; minimum NumPy passed **154** independent workflows.
After correcting a portable test's assumption about FPU trapping, actual pinned
Pyodide passed **134 contracts** and all **13 maintained module suites**.
The original failed WASM test receipt is retained: pinned NumPy reports
`seterr(all='raise')` but does not trap the tested divide-by-zero. Hook behavior
is compared with direct NumPy in the same runtime, with strict CPU trapping
checks retained separately.

A balanced 36-process CPU profile compares released 0.13, the preceding refined
candidate and that correct granular implementation. It retains all samples and
source/resource/harness hashes. The granular version adds substantial reflected
division allocation costs and roughly 5–21% on several reverse workflows in the
float32 series; float64 control variation is also retained without filtering or
claiming a general causal speedup.

The next candidate batches numerical error isolation around the reverse traversal
and restores the mutable caller policy around actual Tensor callbacks. Exact
builtin reflected scalars use private arrays without allocating an intermediate
Tensor, preserving native reciprocal-first values and derivative order. Dynamic
hook registration, changed error handlers, exceptions, public input/output
mutation and saved-state release have independent native comparisons. Its first
focused gate passes **8,401 tests**, two warnings. The complete current-source
gate then passes **16,895 tests**, one optional Docker skip/seven warnings in
163.88 seconds; minimum NumPy passes **156**, pinned WASM **136** and all **13
maintained module suites**. A further balanced 36-process CPU comparison checks
all source/resource/harness hashes and retains every sample independently.

Against the correct granular candidate, scalar reflected division changes
−50.1%/−36.9% and the long affine chain −8.9%/−4.3% (float32/float64). Against
released 0.13, ordinary scalar add/multiply/divide are approximately 8–10% faster,
reflected subtraction about 46% faster, Tensor/Tensor calls cost approximately
2–6% more, and ordinary short reverse calls approximately 5–10% more. Float64
reflected division still costs 24% more than the old direct-division baseline;
that old baseline has incorrect extreme-value semantics. The affine chain is
about 3%/1% faster than 0.13 in this series. These figures are CPU-only and use
process means of the two per-process medians; they are not browser claims.

The [IEEE phase archive](../../packages/paperleet-torch/benchmarks/results/iteration-14-ieee-prototype.json)
retains all 72 new CPU reports alongside the previous 64, strict native and
portable test sources, gate receipts and benchmark sources. Verified patches
reconstruct the granular and earlier refined phase from the batched candidate;
the refined bridge then reconstructs all four historical snapshots exactly.

A further adversarial native comparison confirms a remaining direct-division
VJP issue: the denominator derivative computes `-g*a/(b*b)` instead of native
`-g*((a/b)/b)`. With float64 `a=b=1e200`, the current result is −0 instead of
−1e−200; with `a=b=1e−200`, it is −infinity instead of −1e200. The native
[manual derivative](https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/FunctionsManual.cpp)
confirms the required order. The quotient-first correction now passes **8,324 focused tests**, one warning,
including 100 new native comparisons and one portable workflow. These cover
float32/float64 large/small values, zero and scaled seeds, all three API forms,
constant numerators, broadcasting and reverse-over-forward gradients. The corrected 35-file source then passes **16,996 full tests** (one optional
Docker skip/seven warnings, 163.25 seconds), **157 minimum-NumPy tests**,
**137 pinned WASM contracts**, **13 maintained module suites**, and native
**77 modules / 9,987 test executions / three repeats / 336 caught mutants**.
Chrome 155.0.8059.82 independently passes the same137 contracts and13 selected
reference/mutant/learner cases. Its pure wheel/sdist contain the exact35 modules
and Apache-2.0 resources; the installed wheel passes157 tests under NumPy1.26.4
without native Torch. Wheel size119,765 bytes, SHA-256
`01fb38e6f0ecc9aa2f38ba89764b8bacd55451c8dbf3f1be1252e0a7d50df948`.
Web checks and production build pass. These results precede integration of newer
main changes described below.

## Complete preintegration measurements

All correctness/build producers are observed terminal before sequential timings.
The final frozen protocol retains **40 CPU**, **40 Node/WASM**, **eight fresh
Chrome processes/Workers**, and **eight CPython allocation reports**, each with
source/resource/harness hashes. Each dtype uses baseline→candidate→candidate→
baseline. Five suites cover13 arithmetic, eight selection, seven runtime,
24 construction and five Tensor-hook cases; every timing keeps11 batches and
three warmups, independent value/dtype/gradient checks outside timing, and full
allocation/cleanup inside timing. CPU uses verified single-thread Apple
Accelerate, Python3.12.11 and NumPy2.5.4; pinned WASM uses Pyodide0.29.4/NumPy2.2.5.

Observed Chrome changes relative to released0.13, from the mean of the two
per-process medians (negative means less time):

| Workload | float32 | float64 |
| --- | ---: | ---: |
| Scalar add | −9.0% | −12.8% |
| Scalar multiply | −7.8% | −11.5% |
| Scalar divide | −9.6% | −13.3% |
| Reflected scalar subtraction | −37.1% | −41.6% |
| Reflected scalar division | +11.6% | +9.1% |
| Tensor/Tensor add | +0.9% | +0.9% |
| Tensor/Tensor subtract | +5.1% | +3.0% |
| Tensor/Tensor multiply | +3.3% | +1.7% |
| Tensor/Tensor divide | +3.3% | +0.9% |
| Add backward | +7.1% | +3.2% |
| Multiply backward | +5.3% | +2.8% |
| Divide backward | +10.4% | +7.9% |
|16-stage affine backward | −3.3% | −7.3% |
| Attention backward | +0.5% | −0.5% |
| MLP training step | +6.2% | +4.2% |
| Retained Tensor-hook backward | +9.5% | +9.1% |

Node/WASM shows corresponding scalar add/mul/div reductions of roughly10–13%,
reflected subtraction38–42%, and affine-chain4.6%/7.5%; reflected division costs
6.5%/10.0% more and short reverse passes about3–11% more. The corrected reciprocal
sequence, NumPy policy isolation and quotient-first derivatives have measurable
costs. Factory controls are mostly close to parity; the Chrome float64
4,096-element float-list case costs7.3% more in this series. Selection backward
costs approximately2–6% more. Same-dtype conversion reaches Chrome's timer floor,
so its raw zeros are retained and no percentage is calculated. These are local
ordinary fixtures, not cold-start or complete-application latency claims.

CPU primary samples show unresolved process variation: float32 reflected division
baseline medians are7.6409 and3.5893 microseconds, giving an apparent38.1%
reduction, and attention baselines290.36/239.36 microseconds. A separate
**16-process ABBA CPU followup** preserves the original data and rechecks
arithmetic plus attention, MLP and AdamW. It observes reflected division+2.8%/
+24.5%, scalar add−7.4%/−7.7%, scalar multiply−11.1%/−10.9%, scalar divide−3.7%/
−6.8%, reflected subtraction about−47%, and division backward+11.6%/+14.2%.
The primary float64 scalar multiply is+9.1%, versus−10.9% in the followup; both
remain recorded. No filtering, pooling or stable causal CPU speedup is claimed
from discrepant series. Other primary and followup controls remain in the archive.

The16-stage add/tanh memory fixture retains the same bytes after normal backward
and final release for both versions; batched numerical scoping adds about1KiB to
reverse peak allocation. Where-chain final-release bytes also match; its normal
retained snapshot differs by280 bytes. These are CPython tracemalloc snapshots,
not RSS or browser heap measurements.

The [complete preintegration archive](../../packages/paperleet-torch/benchmarks/results/iteration-14-preintegration.json)
contains every sample and followup, source/test/harness material, notices, native
and distribution receipt references, browser versions and exact-source
reconstruction links. The
[numerical-order record](../../packages/paperleet-torch/benchmarks/results/iteration-14-numerical-order.json)
bridges this source back to all earlier35-file phases, with each reconstruction
verified by SHA-256.

## Integration still required

After these gates, fetched main is `16386436c08cf8201947950e77698f4d8e77c126`.
It contains0.13.1 NumPy contractions for WASM linear pullbacks, additional linear
contracts, updated foundation lessons/assertion diagnostics, full-catalog WASM
CI and community/reading UI changes. All of that work must be preserved. The
measurements above compare the frozen preintegration source with0.13 commit
`594a984d8fed2fbcba933e946e8177922c742871`; they do not validate the combined
source or claim a comparison with0.13.1. Combined-source correctness, standalone
distribution, updated-main performance, exact main CI and deployed asset checks
remain outstanding. Existing published hashed assets must remain byte-identical.

The independent-library/core-coverage objective remains active. No new GPU
backend or PyPI publication is part of this candidate. A possible later CPU
optimization is private in-place quotient VJP workspace reuse; it is an untested
idea and is not included in any timings or current code.

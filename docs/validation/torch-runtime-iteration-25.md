# StrideTorch 0.25 candidate: NaN-aware reductions

This candidate adds real `nansum` and `nanmean` with method/functional entry
points, dtype casts, analytic VJP and float32/float64 JVP including reverse over
the tangent. Core and complete local application integration gates have passed;
exact main adoption, remote Quality and publication remain separately verified.

The implementation uses a boolean mask with NumPy's `sum(where=...)`, and
retains output-sized valid counts for means. Integer-output floating nansum
follows native's `nan_to_num` composition before casting. Complex CPU kernels,
complex autograd, half JVP and output buffers retain explicit errors. Empty
real-to-complex reductions can bypass the native kernel, but do not extend this
library's declared complex boundary.

Ordinary sum/mean share dimension validation. Both require an explicit `dim`
when supplying `keepdim`; the NaN reductions permit omitted dimensions. Bool,
fractional, repeated and out-of-range dimensions preserve native exception
classes and validation order. Common omitted/single-axis cases avoid unnecessary
general sequence parsing. Ordinary mean's validated opmath implementation is
unchanged.

Direct VJP casts at the forward input dtype before multiplying the current
input mask. Nanmean reuses the forward count even when raw backing data changes.
Saved-input in-place changes are guarded; output mutation is allowed. Half
nanmean preserves the native sum-then-divide overflow behavior, including the
finite subnormal derivative when 65,536 half ones overflow the intermediate sum.

JVP uses `where(primal.isnan(), 0, tangent).sum`, so a NaN tangent at a valid
primal position remains NaN. Nanmean's zero primal reverse edge stays exact for
infinite/NaN values and seeds. A separate guarded branch allows gradients
targeting only the tangent to prune the primal guard. Saved graph release and
retained replay are tested. Native 2.14.1's lazy ZeroTensor gradients cannot
accumulate in-place; the replay comparison clears leaf gradients between calls
to isolate graph lifetime from that native representation limitation.

Primary references are PyTorch v2.14.1's
[reductions](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/ReduceOps.cpp#L1215-L1242),
[nanmean composition](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/ReduceOps.cpp#L1411-L1429),
[derivative registration](https://github.com/pytorch/pytorch/blob/v2.14.1/tools/autograd/derivatives.yaml#L1549-L1551)
and [pullback](https://github.com/pytorch/pytorch/blob/v2.14.1/torch/csrc/autograd/FunctionsManual.cpp#L916-L922).
Code is independently implemented and tested in Python/NumPy.

## Evidence and outstanding gates

The 3,024-case native audit matches all 2,982 supported cases, with 42 explicitly
unsupported empty real-to-complex cases retained in the difference record.
Initial and corrected validation-order sources and failure records are preserved.
Native tests cover float16/32/64 casts, IEEE seeds, scalar/empty/multiple-axis
inputs, model dtype rebinding, current storage masks and JVP graph pruning.
Independent tests include directional finite differences and VJP/JVP duality.

Final-source Node and Chrome each pass 806 NumPy-only contracts on Pyodide
0.29.4 / NumPy 2.2.5. The installed wheel passes 818 contracts on Python 3.10.19 /
NumPy 1.26.4 with native Torch absent. All 42 Python modules and both license
resources match the frozen source. Public and private wheels also pass a fresh
NumPy-only coinstallation check: both execute the new reduction and pullback,
and uninstalling either leaves the other's 44 files unchanged.

Final infrastructure passes 26,466 tests (one optional Docker skip, 23 warnings,
181.79s). Balanced CPU/WASM timing completes all 16 fresh sequential children,
using ABBA order for each dtype/backend, 11 batches and three warmups. Timing
compares the old where/sum composition with new direct reductions on
matching finite-gradient fixtures, with ordinary sum/mean/exp controls. It does
not compare unsupported baseline APIs, pathological gradients, native Torch,
cold start, peak memory or application speed.

Small nansum calls take 33–35% less time; small/casted nanmean calls take 51–56%
less. Row nansum ranges from 3–17% less, while row/column nanmean ranges from
31–43% less. Reverse rows improve 5–11% for nansum and 22–32% for nanmean.
The ordinary sum control ranges from unchanged to 5.1% slower; unchanged exp
ranges from 5.3% slower to 2.2% faster, demonstrating shared-host variation.
Mean controls range from unchanged to 8.2% faster. These controls and both
process samples per label are retained; small differences are not causal wins.
CPU environment requests two BLAS threads; actual threading is unverified.
No local tests, builds or audits overlap the controlled timing phase.

[Evidence](../../packages/stridetorch/benchmarks/results/iteration-25.json) and
the [reproducibility archive](../../packages/stridetorch/benchmarks/results/iteration-25-harness.tar.gz)
retain frozen sources, all timing children, original failed test fixtures,
the corrected validation-order audit, distribution artifacts and final gates.

## Complete application integration

Original native CI passes 78 problems / 10,218 executions / 342 mutants / three
repeats in 127.91s, within its unchanged 600s aggregate and per-module T3 limits.
All visualization data regenerates without changing prior immutable resources.
The export preserves all 6,756 published hashes and creates 158 reachable assets.
The autograd profile carries 38 Python modules plus LICENSE/NOTICE, 467,986 raw
bytes / 108,275 gzip-estimated bytes; schedules carries three modules and both
notices, 37,343 raw / 11,550 gzip-estimated bytes. These are payload sizes, not
measured HTTP transfer or startup times.

The complete maintained exported WASM check passes all 78 problems and every
mutant in 383,839ms, with zero calibration/run/total budget violations. The CI
harness now discovers all NumPy-only library suites, excluding only host
launcher/install contracts. Its actual exported attention VM additionally passes
25 suites / 806 tests, with per-suite pass counts and source hashes retained.
The earlier partial run was intentionally stopped before this coverage change;
it is preserved as partial and does not substitute for the final complete run.

All 34 upstream UI/curriculum files through `1305e09d` are preserved byte-for-byte.
Their changes do not touch the 44 library/resource or 161 frozen export hashes.
The full frontend sequence passes with 51 bilingual guides / 153 prompts; the
26 affected Python catalog/publication tests pass. Later company-source changes
pass their four affected flow checks and production build. The actual Pages
repository variable selects the hosted profile, which also builds successfully:
three initial JavaScript assets, 403,136 raw / 130,255 gzip bytes, within unchanged
caps. Full logs retain ephemeral Vite shutdown/HMR diagnostics; all commands and
assertions pass. No application browser visual QA is claimed.

Rebuilt private wheel/sdist contain the exact new core and latest application
catalog/server, with paths disjoint from the standalone wheel.
[Integration evidence](../../packages/stridetorch/benchmarks/results/iteration-25-integration-gates.json)
and its [202-file archive](../../packages/stridetorch/benchmarks/results/iteration-25-integration-gates-harness.tar.gz)
retain reports, harnesses, frozen export bytes and distribution artifacts.

## Main and actual site adoption

Non-force main adoption at `22a0f289` passes exact Quality run 38896970644:
26,466 infrastructure tests (one optional skip, 26 warnings, 401.77s), original
native catalog in 299.51s, and all 78 WASM modules in 983,423ms without budget
excess. Its own Pages trigger is superseded by later main work.

Runtime-identical descendant `90ac44ef` preserves all 44 core/license and 161
frozen export hashes. It additionally preserves upstream parallel quality
checks, including the required infrastructure job. Quality run 38899157716
passes all four required jobs; its complete two-process WASM run passes in
537,730ms without budget excess, with identical per-module inputs/counts/budgets.
These runner durations are not controlled speedup comparisons.

Eligible Pages run 38900518818 completes the actual build and deploy jobs.
Subsequent live verification matches all 78 decoded starters and active manifest
references, all 158 immutable payload bytes and both 0.25 runtime versions and
Apache notices. [Remote adoption evidence](../../packages/stridetorch/benchmarks/results/iteration-25-remote-adoption.json)
and the [actual receipt archive](../../packages/stridetorch/benchmarks/results/iteration-25-remote-adoption-harness.tar.gz)
preserve job steps, equivalence, reports and live byte checks. Independent public
package publication remains separate work.

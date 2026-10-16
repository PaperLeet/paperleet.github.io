# StrideTorch 0.23: bounded public-call validation

This version reduces repeated public-operator dispatch overhead. Numerical
kernels are unchanged from 0.22. Main adopts 0.23 at `3cea3727`, and its exact
Quality run passes all required jobs. Site publication is unconfirmed: later UI
commits supersede its eligible Pages triggers. No skipped run counts as deploy.

## Behavior and memory

Each public operator lazily retains at most 32 successful signature patterns,
keyed by positional arity and keyword-name order after existing unsupported
options are extracted. It never retains argument values, tensor objects,
gradients or execution modes. Excess/unknown patterns use ordinary binding;
invalid bindings are not cached. Values, dtype/dimension validation, unsupported
options and forward/reverse capability checks still execute on every call.
Underlying function exceptions keep their existing behavior.

Only ordinary string keyword names may enter the cache. A keyword-name subclass
can carry an arbitrary payload through attributes; the first prototype retained
such names and failed an explicit lifetime regression. The final implementation
does not retain them. Weak-reference checks cover argument values, nested tensors,
returned tensors and these indirect payloads.

## Correctness so far

Before the latest unrelated server/workspace merge, the complete infrastructure
suite passes 25,612 tests, with one optional Docker skip and 17 warnings (272.71s).
Actual Node/Pyodide 0.29.4/NumPy 2.2.5 and Chrome
155 Worker runs each pass 774 independent contracts. The isolated installed
wheel passes 794 tests on Python 3.10.19/NumPy 1.26.4 without native Torch.
Standalone/private wheels and source archives contain identical 41-module core
and license bytes with disjoint installed paths. Native Torch references remain
part of the full infrastructure suite; standalone tests also use independent
NumPy values, gradients and object lifetimes.

Mixed positional-only, keyword-only, variadic, missing/duplicate/unknown argument
cases revisit valid patterns after errors and with changing values. Warm calls
still recheck unsupported output options, nested differentiable inputs, no-grad
contexts, JVP, dtype changes and invalid dimensions.

The initial native candidate passes 25,611 tests. Initial Node and Chrome runs
each pass 772 tests and fail one fixture: Python 3.13's signature binder accepts
one missing positional-only parameter in a function with variadic keywords,
while the Python function correctly rejects the call. The corrected fixture
checks both binding and actual function behavior, without relaxing product
validation. Initial failures and frozen source/tests remain preserved.

The original native catalog passes all 78 problems, 10,218 executions across
three repetitions and 342 mutants (123.49s), retaining its existing budgets.
Export preserves 6,440 published immutable assets; all 158 new assets are
reachable. After merging upstream curriculum at `db6408fd`, all nine upstream
files match exactly and all core/license hashes and 161 frozen export files
remain unchanged. Fresh private wheels/source archives retain the latest
application catalog metadata and identical standalone core bytes. Repeated
frontend checks validate 48 bilingual guides/144 prompts; build and Pages policy
also pass. Complete exported WASM passes all 78 modules without any browser
budget violations (526.647s aggregate). Flow Matching takes 119.498s calibration,
21.980s learner and 144.720s total, within unchanged 240/60/300s limits. These
application timings are budget evidence, not controlled release comparisons.
Exact main Quality is verified below; final site adoption remains unconfirmed.

Latest upstream `c3f8d5c3` adds local request-error and workspace/draft recovery
fixes. All five changed files are preserved exactly. The affected server suite
passes five tests (5.36s); complete frontend checks, production build (5.46s)
and Pages policy pass again. Refreshed private wheel/source archives include the
latest server and catalog bytes. All 43 core/license hashes and 161 exported
files remain identical to the complete numerical/WASM gates, so those numerical
workloads are not repeated for these application changes. Remote Quality will
execute the complete final merged infrastructure, with results recorded below.

## Controlled dispatch measurements

Three phases retain 48 fresh sequential processes. Each phase uses CPU/WASM,
float32/float64 and baseline/candidate/candidate/baseline ordering. All children
finish successfully, with source/harness hashes checked around every child.
No local tests, builds or audits overlap these timings. The machine is shared;
CPU environment controls request two threads without establishing actual BLAS
threading. Both versions execute identical numerical work and sampling.

The first two phases use 2,000 forward/400 reverse calls per batch. The middle
phase's unchanged direct-method controls drift by as much as 18%; its percentages
are retained without being treated as precise causal gains. The final phase uses
5,000 forward/1,000 reverse calls per batch, eleven batches after three warmups.
Its unchanged controls differ by about 0–6%, so individual small differences
still should not be interpreted as implementation effects.

| Fixed workload | CPU time reduction | WASM time reduction |
| --- | --- | --- |
| Public exponential variants | 22–29% | 3–32% |
| Keyword sum | 27–29% | 23–29% |
| Addition/matrix multiplication | 26–30% | 26–27% |
| Concatenation | 32–35% | 35–36% |
| Paired moments | 14–15% | 11–12% |
| Complete functional reverse | 12–17% | 13–17% |

These are steady-state small-tensor library calls with repeated argument
patterns. They do not measure first binding, import, cold start, native Torch,
peak memory or application speed. One float32 WASM optional-output exponential
case shows only a 3% gain; it is retained rather than averaged away.

The [development evidence](../../packages/stridetorch/benchmarks/results/iteration-23.json)
and [source/test/harness archive](../../packages/stridetorch/benchmarks/results/iteration-23-harness.tar.gz)
retain all phases, failed prototypes, independent distributions and release
snapshots. They explicitly distinguish completed gates from pending adoption.
The [merged gates](../../packages/stridetorch/benchmarks/results/iteration-23-merged-gates.json)
and [merged packaging/frontend archive](../../packages/stridetorch/benchmarks/results/iteration-23-merged-gates-harness.tar.gz)
record the refreshed application metadata and repeated affected checks.
The [complete local release gates](../../packages/stridetorch/benchmarks/results/iteration-23-local-gates.json)
and [final WASM/archive receipts](../../packages/stridetorch/benchmarks/results/iteration-23-local-gates-harness.tar.gz)
close the full exported catalog gate against the frozen source and 161 files.
The [latest upstream gates](../../packages/stridetorch/benchmarks/results/iteration-23-latest-upstream-gates.json)
and [affected-check archive](../../packages/stridetorch/benchmarks/results/iteration-23-latest-upstream-gates-harness.tar.gz)
retain the final application merge, server checks and refreshed packaging.

## Main Quality and superseded publication

[Exact Quality 38884042959](https://github.com/PaperLeet/paperleet.github.io/actions/runs/38884042959)
passes native, browser-wasm and pilot jobs for `3cea3727`. The complete merged
Linux infrastructure passes 25,613 tests, one optional skip and 20 warnings
(386.22s). Original native catalog passes 78 problems, 10,218 executions, 342
mutants and three repetitions (290.68s). Real WASM passes all 78 modules without
browser budget violations (1,009.695s aggregate). These are gate observations,
not controlled comparisons with other releases.

Runtime-identical UI descendant `1e2a8a66` has identical 43 core/license hashes
and 161 export hashes. Its
[Quality 38885685787](https://github.com/PaperLeet/paperleet.github.io/actions/runs/38885685787)
also passes all required jobs, but newer UI work supersedes its Pages trigger.
Both superseded/skipped workflow histories are retained. A mature 0.24 can
supersede an unreleased site version; the actual final deployment still requires
eligible Pages steps and frozen live-file verification.

[Main-quality evidence](../../packages/stridetorch/benchmarks/results/iteration-23-main-quality.json)
and [workflow/artifact archive](../../packages/stridetorch/benchmarks/results/iteration-23-main-quality-harness.tar.gz)
record that main adoption is proven while site publication is not.

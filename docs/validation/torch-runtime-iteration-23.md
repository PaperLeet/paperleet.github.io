# StrideTorch 0.23 candidate: bounded public-call validation

This development version reduces repeated public-operator dispatch overhead.
Numerical kernels are unchanged from 0.22. Main uses 0.22; the 0.23 candidate
has not been integrated or deployed.

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

The final infrastructure suite passes 25,612 tests, with one optional Docker
skip and 17 warnings (272.71s). Actual Node/Pyodide 0.29.4/NumPy 2.2.5 and Chrome
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
also pass. Full exported WASM and main/remote adoption remain pending.

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

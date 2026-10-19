# Iterations 98–104: complete dispatcher timings and broader norm boundaries

The complete-version CPU diagnostic and all queued norm investigations are
terminal. Zero propagation now matches its 576-case native matrix on CPU,
Node/Pyodide and real Chrome. A revised norm HVP rule matches the original
960-case boundary matrix on CPU, but a larger 4,480-case matrix exposes remaining
extreme-float32 differences. Production sources and main remain unchanged.

## Dispatcher behavior and CPU performance

All three fresh complete-version timing processes completed with verified raw
rows, source hashes, paired ratios and Williams controls. Minimum chunks were
97.88, 98.49 and 99.31 ms; maximum A/A deviations were 0.76%, 0.98% and 1.52%.

The candidate registry removes empty per-call option dictionaries and retained
empty registration sets. Its paired median D/B ratios across processes are:

| Linear forward fixture | float32 | float64 |
| --- | ---: | ---: |
| Tiny | 0.97752 | 0.97672 |
| Medium | 0.98405 | 0.97709 |
| Batched | 0.98572 | 0.98855 |

Lower is better. This is a focused twelve-flow CPU diagnostic, not a general
application speed claim. The candidate separately passes all 11,982 existing
SDK tests on CPU, Node/Pyodide and Chrome, with no skips. The WASM tests validate
behavior, not the CPU speed ratios.

The complete 0.78 baseline B does not reproduce the original roughly 6.7%
batched-forward regression under this new protocol. That does not explain the
old result or discard the old 124-flow controls. CPU float32 batched backward
still shows a smaller slowdown; allocating versus reusing bias output also
has a peak-memory tradeoff. Full-workload qualification and policy selection
remain necessary before production adoption.

## Zero rules and rejected higher-derivative hypothesis

Iteration 96 repairs the original 24 chain discrepancies and discovers another
24 among newly added chains. Iteration 99 adds explicit declarations for the
validated view/unsqueeze/squeeze/clone operations, multiplication and scalar
subtraction, preserving reflected subtraction and undeclared numerical rules.
All 576 chains then match both native references. Its 11,982 existing CPU tests
and 95 targeted tests pass; Node and Chrome each pass the combined 12,077 tests
and reproduce the same norm findings. The remaining norm findings are retained,
so those complete test counts do not qualify the whole prototype.

The unchanged parent control demonstrates why aggregate HVP counts are
insufficient. Iteration 97 fixes 28 cases but regresses 20 previously correct
cases in the same 960-case matrix. It is rejected. Its original source, logs,
oracles and paired observations remain preserved.

Successors separate saved-primal and nonlinear-direction routes, distinguish
the singular power interval from the specialized two-norm, and use the norm's
reverse-derived linearization during HVP. Nonfinite reductions retain the
additional derivative of a nonlinear upstream Jacobian. This follows the
native functional HVP's reverse-based construction rather than assuming a
direct forward JVP has the same singular-point behavior.

Iteration 102 matches both native versions on all 3,660 primary norm stages,
all 1,600 original composition stages and all 960 HVP boundary cases. This is
CPU probe evidence; it has not received complete standalone/package/application
or performance qualification. The newer HVP rules have not yet been requalified
in WASM. The zero-rule WASM proof applies to the shared iteration-99 tensor and
graph rules, not to every later norm change.

## Expanded 0.79 candidate and float32 arithmetic

Iteration 103 prepares an isolated 0.79 package candidate combining those norm
rules with the validated registry candidate. Its package and source snapshot
remain outside canonical production code. The HVP matrix expands to 4,480 cases:
two dtypes, eight input families, twenty orders and fourteen expression chains.
Both native references reproduce all 320 original composition HVP controls.

The candidate initially matches 4,426 cases per native version and differs in
54. All differences are float32: 49 at very small magnitudes and five at very
large magnitudes. They include genuine finite/NaN/Inf distinctions and cannot
be removed by simply relaxing numerical tolerances.

Three large-magnitude sine compositions expose an accumulation-type rounding
issue. Native CPU norm casts the order to its accumulation type and computes
the reciprocal in that type, as shown by
[NormOps](https://raw.githubusercontent.com/pytorch/pytorch/v2.6.0/aten/src/ATen/native/SharedReduceOps.h)
and its
[CPU dispatch](https://raw.githubusercontent.com/pytorch/pytorch/v2.6.0/aten/src/ATen/native/cpu/ReduceOpsKernel.cpp).
The old prototype computes the reciprocal as a Python double before rounding.
Its tiny relative norm error becomes a large phase error when composed with sine.

Iteration 104 explicitly computes both order and reciprocal in the accumulator
dtype. It fixes those three exact native cases without introducing a new case
discrepancy in the expanded matrix. The remaining result is 4,429 matches and
51 differences against each native version. Extremely small float32 HVPs still
need investigation of arithmetic ordering, cancellation and under/overflow in
the native reverse construction. The other two large-magnitude findings remain.
No cases have been dropped or relabeled as successes.

## Evidence and remaining qualification

The task evidence preserves:

- Iteration 95's frozen complete-version plan, six receipts and recomputed
  dispatcher observations.
- Iterations 98–102's unchanged parent, rejected hypothesis, zero-rule results,
  native probes and subsequent CPU norm refinements.
- Iteration 100's hash-bound source/tests and four successful actual WASM runs,
  including exact test counts and retained numerical findings.
- Iteration 103's isolated 0.79 package/source, expanded native references and
  54-case findings; iteration 104's rounding revision and 51-case findings.

All earlier frozen experiment inputs were rechecked unchanged. The latest
rounding revision also rechecks its own immutable sources before and after
execution. All owned jobs in this checkpoint terminated normally. No source
integration, push, GitHub CI, tag, release, publication or deployment occurred.

Repair the remaining float32 cases, recheck the original probes and paired
controls, qualify the selected combined source on CPU/minimum NumPy/Node/Chrome
and standalone packaging, then renew full application, export, lifetime and
performance checks. Current-main reconciliation and fresh guarded adoption
remain required. The continuing core/API/CPU/WASM/WebGPU objective stays active.

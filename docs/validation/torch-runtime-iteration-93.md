# Iteration 93: logical-zero graph prototype and complete CPU factor observations

The continuing goal remains active. A new graph prototype addresses one source
of norm-composition discrepancies, but has not run its numerical qualification.
The three CPU factor processes are complete and independently verified; actual
Node and Chrome factor timing still follows. Production's 63 resources and
execution HEAD are unchanged. No main adoption, CI, tag, publication or
deployment occurred.

## Logical zero cotangents

The norm composition findings in [iteration 91](torch-runtime-iteration-91.md)
include a distinction between native logical-zero cotangents and ordinary
NumPy arrays containing zero. A constant-zero derivative can propagate through
an upstream infinite value without running zero-times-infinity arithmetic.
Ordinary numerical zero arrays must retain their existing IEEE behavior.

The isolated `torch-iteration-93` overlay introduces an internal, storage-free
logical-zero marker in `_graph.py`. Only the norm sign rule's known constant
derivative emits it. The engine keeps a real zero gradient distinct from an
unused edge, merges it with real contributions, and materializes arrays when
returning or accumulating observable gradients. Target pruning, saved-version
checks, callbacks and graph release still execute. Identity/None intermediate
hooks preserve the marker if the supplied storage was not changed; replacement
gradients follow the numerical path. Further hook semantics remain part of
qualification, not an assumed native-compatibility claim.

Ten stdlib-only checks exercise the actual graph control flow with small
stand-ins, without importing NumPy during live timing. They cover deferred
materialization, numerical-kernel bypass, both accumulation orders, joint-parent
dtype handling, saved versions, retained gradients, callbacks, graph retention
and final release. These metadata checks pass. They do not establish numerical
compatibility or prove that any of the 20 composition discrepancies is fixed.

A frozen qualification supervisor waits for the known factor supervisor to
terminate. It will then run the 780 isolated norm cases, 320 nonlinear
composition cases, 93 targeted tests, the 11,982 existing SDK contracts on CPU,
and the combined 12,075 existing/new tests in actual Node/Pyodide and Chrome.
Counts and failures are checked explicitly. All discrepancies remain visible;
the queue does not declare success by excluding difficult cases. If the timing
supervisor fails, it records the failed prerequisite and runs no numerical job.

The earlier iteration-91 numerical evidence applies to that earlier graph.
The changed graph is unqualified until its own queued checks complete. Full
package, application and performance qualification remain separate requirements.

## Three completed CPU factor processes

The independent verifier matches all 54 completed flow rows to their durable
raw records and successful process receipts. It recomputes every ratio, checks
the equal iteration counts and Williams position/ordered-adjacency counts, and
checks GC-matched preflight and measured-chunk minima.

| CPU process | Minimum measured chunk | Maximum A/A drift | A/A cases above 5% |
| --- | ---: | ---: | ---: |
| 0 | 90.51 ms | 2.71% | 0 |
| 1 | 85.14 ms | 2.24% | 0 |
| 2 | 76.14 ms | 4.68% | 0 |

| Batched forward, median across processes | Current B / A0 | Out-of-place fast C / A0 | Raw paired B / C |
| --- | ---: | ---: | ---: |
| float32 | 0.91354 | 0.91651 | 0.99164 |
| float64 | 0.95443 | 0.95238 | 0.99989 |

Lower is better. Paired B/C is recomputed from raw round ratios before
aggregation, not divided from the displayed aggregate medians. These
observations do not support selecting an out-of-place bias change as the fix
for the original full-version slowdown.

## Comparison scope and next diagnosis

The factor baseline uses the 0.77 `_nn.py` inside an otherwise unchanged 0.78
runtime. The original 124-flow campaign compared the entire 0.77 package with
the entire 0.78 package. The unchanged factor remainder therefore does not
isolate all differences between those full versions. Static inspection confirms
the same legacy `_nn.py`, a changed general operator registry, and no bytecode
cache files in the inspected original/factor source trees. The registry is an
unisolated variable, not an established cause of the slowdown.

The repeated original CPU float32 slowdown of roughly 6.7% and six original
A/A anomalies remain open. The factor results neither erase that evidence nor
qualify the full version for adoption. Finish the actual WASM factor timings,
then distinguish full-package/dispatch effects from the Linear implementation
under matched measurements. The production choice still needs renewed broad
performance evidence.

## Evidence and remaining work

The task's `torch-iteration-93` directory contains the logical-zero overlay,
metadata checks, frozen qualification inputs and its waiting supervisor. It
also contains the independently verified three-process CPU factor observations;
the earlier first-two-process observation is preserved separately. Raw factor
results and the continuing supervisor remain in `torch-iteration-92`.

Resolve the norm composition and any new graph findings without narrowing the
contracts. Complete the factor investigation and explain the full-version
performance regression before selecting source changes. Requalify the selected
library/package/application inputs, reconcile current main and use fresh guarded
adoption without CI dispatch or release. Core API coverage and CPU/WASM/WebGPU
efficiency work continue afterward.

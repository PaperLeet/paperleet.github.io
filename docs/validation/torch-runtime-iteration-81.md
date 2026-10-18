# Iteration 81: partial forward AD and complete library qualification

This continues the unreleased 0.78 candidate from
`d3427ea2f3683cb3c0fc3b938d2bc1e6a102f2dd`. The continuing library-efficiency
and core-API goal remains active. Main adoption, GitHub CI, tags, publication and
deployment have not been performed by this iteration.

## Correctness changes

The broader mixed-input controls exposed cases absent from the first affine
tests, which supplied tangents for all three operands:

- An integer/bool affine result now raises `RuntimeError` when an active
  floating input would require a gradient. A `no_grad` inference call remains
  available. This matches the native reference instead of silently discarding
  the floating input's gradient requirement.
- Partial direct JVP accounts for undefined native ZeroTensors in its broadcast
  shape and promoted dtype. It retains zero reverse edges to matrix factors
  without evaluating their values or constructing physical zero factors.
- An already wider tangent survives a second affine operation. The full-input
  JVP path avoids an unnecessary partial-tangent dtype plan.
- Empty matmul JVP reuses the original forward-selected kernel for derivative
  dtype rules. Re-dispatching after unwrapping a dual had exposed reverse flags
  that incorrectly switched valid mixed-dtype empty bmm calls to dtype-strict
  folded mm. Strict mm/dot tangent dtype checks are retained.

There are 31 additional public regression cases beyond iteration 80. They cover
partial affine derivatives, explicit zero reverse gradients, shared captured
operands, dtype promotion, chained affine operations, integer outputs and all 16
empty-matmul cases found by the refreshed native controls.

## Verified library scope

| Check | Result |
| --- | --- |
| Canonical CPU SDK, Python 3.12.11 / NumPy 2.5.4 | 11,995 passed, one warning |
| Isolated installed wheel, Python 3.10.19 / NumPy 1.26.4 | 11,992 passed, three skips |
| Node / Pyodide 0.29.4 complete core SDK | 11,982 passed |
| Real Chrome 155.0.8059.82 / Pyodide complete core SDK | 11,982 passed |
| Refreshed older native matrix controls, each WASM runtime | 11,341 cases; no unexplained differences |
| New mixed affine controls, CPU and each WASM runtime | 58,320 cases; 55,770 matched, 2,550 existing AD boundaries |
| Public/private wheels and both sdist rebuilds | All 63 runtime resources match the frozen source hashes |
| Installed namespace isolation and coinstallation | Passed; standalone installation requires no native Torch or PaperLeet |

The older native controls retain 180 unsupported nonleaf mutations, 79 complex
autograd boundaries and three unsupported output buffers. The mixed affine
suite retains its explicit complex/float16 AD limits, including rejection-family
differences. These counts do not assert unrestricted native Torch equivalence.
The pinned numerical/dispatch reference is Torch 2.6.0; the development native
environment also carries Torch 2.14.1. Empty derivative layout observations are
retained separately from mathematical/AD-flag comparisons.

The first full browser runs passed all SDK tests but failed 16 native matrix AD
controls. Those complete failing results and their source snapshot remain in
`attempt-2-empty-matmul-dispatch`; the corrected full runs are separate evidence.
Earlier immutable iteration records have not been rewritten. The refreshed old
controls materialized zero native ZeroTensors, so their old observation path did
not encounter the unallocated-storage issue found in iteration 80's scalar probe.

Previous task environments were removed outside this iteration while work was
in progress. New development and native-reference environments were created
under the iteration 81 directory. Their interpreter, NumPy and Torch paths are
recorded and point into that directory; they do not depend on the primary
checkout's environment. The primary checkout and its environments were not
modified.

## Performance and remaining adoption gates

The prior long-block measurements showed all implementations slowing together
within a flow. The new CPU experiment shares operand and gradient buffers,
interleaves short measured chunks, rotates label order, and retains wall time,
process time and every sample. Across 26 flows, three fresh processes and three
batches of nine interleaved rounds, the minimum chunk exceeded 42 ms. None of
the 78 A/A flow/process controls exceeded 5% drift; maximum drift was 2.11%.

Small Linear inference elapsed-time ratios versus 0.77 were 0.838 (float32) and
0.837 (float64); medium Linear inference ratios were 0.848 and 0.873. Empty mv
backward ratios were 0.809 and 0.808. Medium Linear backward was essentially
unchanged at 0.999 and 0.998. The new affine APIs were 0.729–0.938 of the
equivalent baseline composition's backward time, depending on shape and dtype;
this compares a new fused API with an existing composition, not an older fused
API. These are targeted CPU observations with the full raw paired samples
retained. They do not establish whole-application or browser speedups.

Before main adoption, complete broader CPU/WASM performance and lifetime
qualification, reconcile current main, run the full application CPU suite and
three native catalog repetitions, regenerate judge exports, satisfy whole-catalog
WASM budgets, and run frontend checks, both build profiles, worker/asset checks
and a fresh nested-route browser test. Finish the explicit evidence archive and
use the established guarded main-push procedure. Further core API work and an
evidence-based WebGPU routing decision remain part of the continuing goal.

Reproducible programs, complete logs, raw controls, wheel receipts and source
freezes are in the task's `torch-iteration-81` evaluation directory. This is a
development qualification record, not a release or main-adoption assertion.

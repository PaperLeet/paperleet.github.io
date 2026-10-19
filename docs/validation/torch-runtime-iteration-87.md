# Iteration 87: Node repeat and improved unexecuted factor scheduling

The original 0.78 timing campaign continues on the same frozen candidate.
The second fresh Node/WASM process has completed all 124 flows. No runtime
implementation has been selected or changed, and no main adoption, GitHub CI,
tag, publication or deployment was initiated. The independent-library goal
remains active.

## Second Node observation

The complete JSON matches every flushed raw flow record, source identities and
the successful subprocess/log receipt. Minimum chunk time is 23.38 ms.
Five A/A controls exceed 5%, with maximum drift 10.62%: float64 scores256
forward, float32 tiny-SDPA forward, float32 vector-batch forward/backward, and
float64 affine-suite empty-mv backward. These remain visible; this process is
not treated as a clean performance qualification.

Batched Linear remains different from the
[three-process CPU observation](torch-runtime-iteration-85.md):

| Second Node batched Linear forward | B/A0 | A1/A0 | Paired B/A1 |
| --- | ---: | ---: | ---: |
| float32 | 0.98635 | 0.97599 | 1.02056 |
| float64 | 0.99814 | 0.99947 | 0.99461 |

The sensitivity metric is recomputed within raw rounds before taking batch and
process medians. Neither completed Node process reproduces the roughly 6.75%
CPU float32 slowdown in this fixture, but the remaining Node and Chrome
processes and the control review are still required. Different runtime builds
and execution paths do not establish a causal explanation by themselves.

## Factor scheduling refinement before execution

Inspection found that the prepared five-label factor experiment balanced
positions through cyclic rotations but not the preceding label's distribution.
Before any of those numerical jobs ran, the two idle waiting supervisors were
stopped and their original sources, plans, logs and progress snapshots retained.
The original 124-flow measurement supervisor and all its children were left
running. These cancellations were a pre-execution method refinement, not an
observation-timeout restart or removal of inconvenient measured data.

Versioned v2 programs use the same four source variants and the same 18 flows.
Each batch has ten orders: every label occupies each position twice, and every
distinct ordered adjacent pair occurs twice within the rounds. Batch order is
seeded and shuffled. Static checks verify these invariants for 100 seeds, and
the result verifiers check them again in measured raw records. This balances
within-round first-order carryover; it does not claim to eliminate cross-round
boundaries or arbitrary longer-lived cache/allocator effects.

Three batches use a nominal 75 ms calibrated chunk for the fastest label, with
the existing 40 ms hard minimum. Equal iteration counts, shared operands/seeds,
fresh processes and durable per-flow raw records are retained. Numerical
validation still precedes timing. The new programs have passed syntax, source
identity and static schedule checks; they have not yet executed numerical or
timing jobs.

The v2 CPU factor supervisor waits for all nine original CPU/Node/Chrome timing
jobs to finish and verifies their receipts. The v2 WASM factor supervisor then
waits for that actual CPU factor process and requires its completed receipts
before running either WASM runtime. All original v1 plans/programs are unchanged
and their old progress records explicitly mark pre-execution cancellation.

## Evidence and remaining work

The task's `torch-iteration-87` directory contains the two-Node verification,
pre-execution snapshots, schedule verifier, frozen preparation receipt and
current queue identities. Versioned CPU factor inputs/programs remain under
`torch-iteration-85`; versioned Node/Chrome bridges remain under
`torch-iteration-86`. The original measured sources and results remain under
`torch-iteration-82`.

Complete and review the running measurement campaign, execute the balanced
factor experiment, resolve material slowdowns and control excesses, and qualify
the selected runtime change. Reconcile current main and verify its UI/build/
worker/page inputs and explicit evidence archive before guarded adoption.
Further core-API and backend efficiency work continues afterward.

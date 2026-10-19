# Iteration 88: first real Chrome timing process

The unchanged 0.78 candidate has completed the first fresh real Chrome process
of the 124-flow campaign. Three CPU and three Node/WASM processes are already
complete; two Chrome processes remain. The continuing library goal stays active.
No main adoption, GitHub CI, tag, publication or deployment was initiated.

## Verified Chrome observations

The separate verifier matches all 124 flushed raw flow records to the final
JSON and successful process receipt, checks candidate/baseline source identities,
equal iteration counts and label orders, and recomputes every round, batch and
process ratio. The minimum recorded chunk is 39.90 ms. Maximum A/A drift is
4.03%; no control exceeds 5% and no flow exceeds the automatic 10% slowdown alert.
These are first-process observations, not a complete performance qualification.

| First Chrome batched Linear forward | Candidate / baseline | A1 / A0 |
| --- | ---: | ---: |
| float32 | 0.99399 | 0.99600 |
| float64 | 0.99601 | 1.00000 |

This process does not reproduce the roughly 6.75% CPU float32 slowdown.
The CPU regression, its separate empty-mv control excess, and the five Node
control excesses remain open. Backend/runtime differences and incomplete
Chrome sampling do not identify the underlying cause or justify shipping the
candidate unchanged.

## Independent preparation during the live measurements

The [balanced factor experiment](torch-runtime-iteration-87.md) remains queued
behind the original campaign. Its CPU and WASM supervisors are live and have
executed no numerical jobs. Production sources and running benchmark inputs
remain unchanged; no owned heavy diagnostic job overlaps the timing process.

The existing sixteen pure release-chain contract cases also pass in a separate
evidence directory. All seven canonical policy files and the mandatory ordinary
push wrapper retain their expected hashes. This checks local timestamp/chain
logic only: no remote adoption audit or push was run. Historical copied guard
provenance describes the earlier 62-resource runtime, so the guard-contract
directory is explicitly not prepared release input for the current 63-resource
candidate. Actual adoption still needs current inputs and a fresh remote-chain
audit through the unchanged wrapper.

## Evidence and next actions

The task's `torch-iteration-88` directory contains the first Chrome verifier,
result and frozen process receipt, plus the separate `guard-contracts` results
and scope record. Completed raw Chrome samples remain in `torch-iteration-82`.

Finish the original Chrome measurements, run the balanced CPU/Node/Chrome
factor experiment, resolve performance/control issues, and qualify the selected
runtime change. Reconcile then-current main, verify UI/build/worker/page inputs
and the explicit archive, and use the guarded ordinary push only after that
evidence is complete. Core-API and backend efficiency work continues afterward.

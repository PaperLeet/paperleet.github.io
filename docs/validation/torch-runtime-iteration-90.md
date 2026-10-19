# Iteration 90: complete paired campaign and norm WASM validation entry points

All nine fresh processes of the unchanged 0.78 performance campaign are now
terminal and successful: three CPU, three Node/Pyodide and three real Chrome
processes, each with 124 flows. Independent verification recomputes every
round, batch and process ratio, checks equal iteration counts and label position
counts, matches durable raw records, verifies source identities and successful
process receipts, and agrees with the original supervisor's summary.

The overall outcome is **review required**, not performance qualification.
The continuing goal stays active. No production source, main adoption, GitHub CI,
tag, publication or deployment changed.

## Measurements that determine the next action

Ratios below are median elapsed-time ratios across the three fresh processes;
lower is better. The second-baseline result is recomputed from raw paired round
ratios before aggregation, not obtained by dividing aggregate medians.

| Batched Linear forward | B / A0 | B / A1, paired | A/A observations above 5%, all 124 flows |
| --- | ---: | ---: | ---: |
| CPU float32 | 1.06749 | 1.06584 | 1 |
| CPU float64 | 1.03947 | 1.04529 | same CPU campaign |
| Node float32 | 1.00005 | 1.02056 | 5 |
| Node float64 | 0.99179 | 0.99449 | same Node campaign |
| Chrome float32 | 0.99399 | 0.99598 | 0 |
| Chrome float64 | 0.98671 | 0.99407 | same Chrome campaign |

The CPU float32 regression repeats in all three processes, with B/A0 ratios
1.07025, 1.06684 and 1.06749 and controls near 1. Its roughly 6.7% slowdown
requires investigation even though it falls below the automatic 10% alert.
The single CPU empty-mv control and five Node controls remain open. Browser
observations do not establish the CPU cause or erase these control issues.

Other measured improvements remain useful evidence: tiny Linear forward is
roughly 17–19% faster across the runtime/dtype fixtures, and float32 medium
fused affine matrix backward is roughly 9–13% faster than equivalent baseline
composition. These are specific fixtures, not an overall performance claim.
The minimum timing chunks are 31.40 ms CPU, 23.38 ms Node and 32.00 ms Chrome.

The balanced Linear factor experiment has now started. Its validation passed
all 18 geometries/modes with shared operand storage for A0/A1 (legacy), B
(in-place bias/fast return), C (out-of-place bias/fast return) and D
(in-place bias/legacy graph setup). The first CPU timing process is live;
complete CPU and WASM factor results are still needed before selecting a fix.

## Actual WASM validation prepared for the isolated norm

Node/Pyodide and real Chrome Worker entry points now run the exact frozen
780-case vector-norm probe, both native-version comparisons and independent
tests from [iteration 89](torch-runtime-iteration-89.md). They preserve full
probe observations, discrepancies and test failures. Input bytes are verified
before mounting; browser results include the actual browser/Pyodide identity.
All three JavaScript entry points and both new Python programs pass syntax
checks. No numerical execution of these norm checks has occurred yet.

The new supervisor waits for the known CPU prototype supervisor. It runs only
after that supervisor has terminated with successful native comparisons and
tests; if CPU repair is required, it records that prerequisite and executes no
WASM numerical job. CPU prototype validation itself remains queued behind the
factor timing campaign. Thus no norm probe or native Torch import overlaps the
live latency measurements. Production sources and previously frozen inputs stay
unchanged.

## Evidence and remaining work

The task's `torch-iteration-90` directory contains the complete campaign
verifier/result and frozen WASM drivers/plan. Original observations remain in
`torch-iteration-82`. The first CPU factor validation/result/logs remain in
`torch-iteration-85`; its WASM counterpart follows in `torch-iteration-86`.

Resolve the Linear regression and control anomalies from complete factor
evidence. Repair and qualify vector_norm, including both actual WASM runtimes.
Requalify any selected production change through library, native, packaging,
application and performance checks. Then reconcile current main and complete
fresh guarded adoption without CI dispatch or release. Core coverage and
CPU/WASM/WebGPU efficiency work continue after that milestone.

# Iteration 86: first WASM timing process and cross-runtime Linear factors

The unchanged 0.78 candidate has completed its first fresh Node/WASM timing
process. The continuing goal remains active. The CPU regressions and isolated
control excess from [iteration 85](torch-runtime-iteration-85.md) remain open;
this checkpoint selects no runtime change or main adoption. No GitHub CI,
tag, publication or deployment was initiated.

## First Node/WASM process

All 124 flows completed on Pyodide 0.29.4 / NumPy 2.2.5. A separate verifier
matches raw flushed flow records against the complete JSON, recomputes every
round/batch/process ratio, and checks source identity, equal iteration counts,
label orders and the successful subprocess receipt.

The minimum recorded chunk is 32.27 ms. Maximum A/A drift is 4.80%; none of the
124 controls exceed 5%, and no flow exceeds the automatic 10% slowdown alert.
Remaining Node and real Chrome processes are still required. These are warm
Python/WASM operation timings, not application throughput or GPU measurements.

| Batched Linear forward | First Node B/A0 | First Node A1/A0 | Three-process CPU median B/A0 |
| --- | ---: | ---: | ---: |
| float32 | 1.00005 | 0.99181 | 1.06749 |
| float64 | 0.99179 | 0.99686 | 1.03947 |

The first Node process does not reproduce the CPU slowdown in this fixture.
Different NumPy builds and backend paths are involved, so this is not a causal
attribution to one implementation detail or a direct CPU-versus-browser speed
comparison. It supports measuring the proposed Linear variants independently
on each runtime before choosing a policy.

## The same factor experiment on WASM

The four [prepared Linear variants](torch-runtime-iteration-85.md) and exact
18-flow Python program are now also frozen for Node/WASM and real Chrome/WASM.
The bridge verifies every source hash before placing files in the Pyodide
filesystem. It also verifies the original factor-plan bytes, preserving all
source/label bindings rather than generating a different browser experiment.

Each runtime first validates all five labels' outputs and gradients. Three
fresh timing processes then use the same balanced five-label rotation and
longer sampling method as the CPU factor test. Completed raw flow samples are
flushed outside measurement intervals. The browser worker has no observation
deadline that terminates a running measurement; completion/error controls its
lifetime.

The WASM factor runner is queued behind the known CPU factor supervisor, which
is itself waiting for all nine original CPU/Node/Chrome jobs. Each boundary
requires the previous process to be terminal and its completion receipts and
log/result hashes to match. These are live verified queues, not executed or
qualified variant measurements. No heavy numerical or GPU job overlaps an
owned timing job, and the canonical 0.78 sources remain unchanged.

## Evidence and remaining work

The task's `torch-iteration-86` directory contains the first Node verifier,
frozen receipt, `first-node-verification.json`, the source-hashed WASM factor
bridge and queued runner. The original completed Node result and raw samples
remain in `torch-iteration-82`; the four private source variants and shared
Python experiment remain in `torch-iteration-85`.

Complete the original measurement campaign, run the CPU and WASM factor
experiments, review controls, and select a supported improvement that resolves
the CPU regression. Qualify that source before reconciling current main,
verifying the final evidence archive and using the established guarded push.
Core API coverage and evidence-based backend work continue beyond that adoption.

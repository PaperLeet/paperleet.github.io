# Iteration 84: first broad CPU process and backend review

This continues the [qualified application and pre-latency stage](torch-runtime-iteration-83.md).
The candidate runtime is still 0.78.0 at execution head
`4cde6dd1045d2097206b18189d65e1c300b0ad5b`; its source has not changed during
the timing campaign. The independent-library goal remains active. This is not
a complete performance qualification or main-adoption record. No CI, tag,
publication or deployment was initiated.

## First fresh CPU process

All 124 flows completed. A separate verifier recomputes every round, batch and
process ratio from the raw elapsed times, checks equal iteration counts and
complete A0/A1/B orders, and matches every flushed raw flow record against the
final JSON. Candidate/baseline source hashes match the validated fixture.

The minimum recorded chunk is 31.40 ms. Maximum A/A drift is 4.68%; none of this
process's 124 controls exceed 5%. No flow exceeds the driver's 10% slowdown
alert. These observations describe only the first of three planned CPU
processes. They establish neither three-process repeatability nor WASM speed.

There is a smaller signal that still requires review:

| Batched Linear forward | Candidate / baseline | A1 / A0 |
| --- | ---: | ---: |
| float32 | 1.0703 | 1.0011 |
| float64 | 1.0395 | 1.0003 |

The float32 case is about 7% slower in this process with a stable A/A control.
The automatic 10% alert is a triage threshold, not an acceptance rule. Earlier
targeted measurements of the same sources showed a gain, so preserve both
observations and examine the remaining fresh processes. Do not combine the
different campaigns into one synthetic ratio or dismiss the new result.

If the slowdown persists, compare private variants separating the in-place
bias addition from the inference AD fast return, with the same operands,
gradient seeds and paired controls. Any selected runtime change needs its own
numerical, packaging, application and performance qualification before main
adoption. The running campaign continues unchanged.

## Backend selection

The [0.78 backend review](torch-backend-routing-078.md) checks the existing GPU
measurements against their original machine records and benchmark source.
Their CPU context is raw NumPy/WASM, predating today's optimized public Tensor
matmul path. Those historical GPU ratios cannot select a current automatic
routing threshold. The current default remains CPU/WASM; the independent GPU
service remains an experiment pending fresh whole-workload comparison and
actual Tensor/storage/autograd and judge integration.

The review also checks the existing program work limit: a 512-square forward
and both matrix gradients require 402,653,184 MAC terms, exceeding the default
268,435,456 aggregate program limit. A fresh comparison must include
partitioning/ownership costs rather than silently increasing that limit. No
GPU job overlaps the current latency measurements.

## Evidence and remaining work

The first-process verifier, frozen receipt, explicit batched-Linear follow-up
and backend evidence hashes are in `torch-iteration-84`. The completed raw
CPU process, input plans and continuing sequential runner remain in
`torch-iteration-82`. Completed results are immutable; live process observations
are dated observations rather than completion evidence.

Finish all three CPU processes and the three fresh Node/WASM and Chrome/WASM
processes. Review A/A drift and material slowdowns, resolve the Linear signal,
reconcile current main's UI with fresh UI/build/worker/page checks, and verify
the evidence archive before guarded main adoption. Broader core-API coverage
and evidence-based backend development remain part of the continuing goal.

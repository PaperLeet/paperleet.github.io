# Cumulative-product execution development

This is a historical development snapshot. The final version bump, full SDK,
application/package/browser qualification and restoration checks are recorded
in [TorchWasm 0.91 qualification](torch-runtime-0.91-qualification.md).

Commit `5335dcda360b561441ce2d5b83fed4e5a712925b` reduces `cumprod` scan
setup and copying relative to the preceding API development at
`56f95bcc9`. This changes only `_product_scan`; all other reduction helper
ASTs and public source resources retain their identities. The
[API/derivative record](torch-cumprod-development.md) remains the baseline.

Eligible scans now use one accumulation directly. Widened scans use this
route only when their accumulator fits the existing 256 KiB planning budget;
larger widened scans retain the blocked carry path. Non-widened output does
not need a separate copy from a workspace into the result. Input conversion
can still require an additional array, and the planning budget is not a
bound on total operation memory.

Medium widened scans explicitly create an opmath buffer and accumulate in it,
avoiding NumPy's extra conversion temporary. Long non-widened real scans
also accumulate in their fresh result buffer: a controlled WASM experiment
found this faster than accumulating directly into a separate output from the
input. Integer scans retain their direct route. Complex scans keep the
observable initial multiply by one. Requested dtype conversion still precedes
wider accumulation; output ownership, contiguous layout and signed-zero
behavior remain intact.

## Numerical validation

The final 656-case module passes on modern/minimum NumPy, Node/Pyodide and
a Chrome Worker. It includes 162 new geometric-prefix/layout cases around
the old short-scan and accumulator-workspace boundaries, plus two closed-form
66-element zero-safe HVP checks. The preceding 492 API/AD cases remain intact.
All 101 other SDK modules are unchanged; this is an affected-module rerun,
not a claim that the entire SDK was executed again.

An additional 696-case matrix covers long and medium scans, float16/32/64,
complex64/128, integer overflow, requested output types, C/F/strided/read-only
broadcast inputs, multiple axes, zeros, infinities, NaNs and ordinary VJPs.
Both native environments pass the numerical/sign comparisons. The modern
native reference matches all 696 complete canonical outputs exactly.
Minimum NumPy retains 72 complex128 forward differences within the recorded
tolerance; those outputs are unchanged from the earlier implementation.

The complete canonical output hashes are unchanged in both CPU environments.
Independent baseline/candidate Pyodide runtimes also match on all 696 cases,
in both Node and Chrome. NaN component signs/payloads are canonicalized;
other values and signs remain part of the hash. Isolated offline exports,
ordinary `import torch` aliases, JVP and HVP smokes also pass.

## Complete-flow observations

The latency corpus has 132 cases with four ABBA observations per case and
environment. The original 36 flat fixtures retain input copies. An additional
96 matrix fixtures preserve C/F/strided inputs and cover both axes of
`(8, 513)` and `(513, 8)`. Cases use float32/64, all-one data with zero-factor
patterns, and inference or square/sum/backward. CPU environments execute
sequentially; Node uses independent baseline/candidate runtimes. The raw
timings and every per-case range are retained.

Ratios below are median candidate/baseline latency, relative to the preceding
API development. Values below one are faster; these are scoped measurements,
not universal speed guarantees.

| Elements | Flow | Modern NumPy | Minimum NumPy | Node/WASM |
| --- | --- | ---: | ---: | ---: |
| 16 | Inference | 1.007 | 0.995 | 1.034 |
| 16 | Reverse | 1.023 | 0.977 | 1.060 |
| 4,096 | Inference | 0.760 | 0.883 | 0.597 |
| 4,096 | Reverse | 0.888 | 0.928 | 0.881 |
| 4,104, matrix layouts | Inference | 0.734 | 0.799 | 0.651 |
| 4,104, matrix layouts | Reverse | 0.921 | 0.914 | 0.908 |
| 65,536 | Inference | 1.040 | 1.092 | 0.903 |
| 65,536 | Reverse | 0.964 | 1.010 | 1.036 |

Medium and matrix flows improve in these groups. Small flows are mostly near
parity, with remaining overhead in some cases. Large CPU inference remains a
cost: its median rises about 4% on modern NumPy and 9% on the minimum
environment. The modern large-inference range includes one 1.316 ratio.
These observations are not discarded or described as an overall speedup.

Native allocation checks use `tracemalloc`, three repetitions after warmup,
with a control confirming NumPy data buffers are visible. They include API
construction, an input copy and operation temporaries; preexisting user data
is excluded. For float64 input/output, median inference peaks fall from about
103 KB to 68 KB at 4,096 elements and from 1.315 MB to 1.051 MB at 65,536
elements, across both environments. Medium float32 peaks are slightly lower;
large float32 controls are unchanged. Reverse-flow peaks are mostly near
parity. This does not measure RSS, total native allocations or WASM heap use.

## Retained development observations and status

The first direct-accumulation route slowed long float64 WASM inference by
about 17%. The isolated copied-in-place control reduced that cost, and the
final results above retain the complete comparison. Another intermediate
route raised medium float32 traced peak allocation by about 20%; the explicit
opmath buffer removes that conversion temporary. Both intermediate results
remain archived alongside the final measurements.

A source update failed under transient ENOSPC. Its hash matched the frozen
preceding source, so no partial source edit was present. Verified hardlink
deduplication of 300 identical frozen archive copies preserved all paths,
bytes and modes and released about 926 MB of logical duplicates. No
measurements or historical records were deleted.

The [machine record](../../packages/torchwasm/benchmarks/results/cumprod-scan-execution.json)
binds exact current source, the parent API proof, all numerical checks,
intermediate/final latency data and native allocation controls. Its
1,165,565-byte archive recovers 2,092 logical files from 333 unique blobs;
SHA-256:
`4e936ed5bb8124a1d325d94d3cd251adde04603417d4d41f2c688e322d3c5d07`.
The public verifier passes evidence-integrity and current-source checks.

Package metadata is still 0.90 during development. The next qualification
needs a version bump, fresh wheels/sdists and installed-package checks,
full native application and all-module WASM runs, fresh immutable exports,
current-main integration and both builds plus actual nested Worker checks.
The guarded Git delivery issue remains unresolved; no previous consumed
grant is reusable. No main push, CI dispatch, tag, publication or deployment
was initiated in this iteration.

# Iteration 83: complete 0.78 application and pre-latency qualification

This continues the [iteration 82 checkpoint](torch-runtime-iteration-82.md)
with the same 63 runtime resources qualified in
[iteration 81](torch-runtime-iteration-81.md). The execution head is
`4cde6dd1045d2097206b18189d65e1c300b0ad5b`, based on main
`077fa5bc6adbea0ed9c84a73fc9272dc09a0cd26`. No main adoption, GitHub CI,
tag, publication or deployment has been performed. The continuing goal remains
active, and the package name remains `stridetorch`.

## Complete application checks

| Check | Result |
| --- | --- |
| Full CPU application and SDK suite | 45,624 passed, one skipped, 34 warnings |
| Native catalog | 89 modules, 12,360 test records across three repetitions; passed in 247.61 seconds within the 600-second budget |
| Whole real Pyodide/WASM catalog | 89/89 modules, 406 mutants, no browser-budget excess; concurrency one |
| Existing immutable assets | All 15,499 retained with matching hashes |
| New exports | 180 reachable assets; 183 live asset/index identities |
| Full frontend checks and four additional incoming UI checks | Passed |
| Default and actual Pages-profile builds | Passed |
| Pages policy, built assets and retained production assets | Passed |
| Exact worker regeneration | Passed; six source inputs match the generated worker bytes |
| Fresh Chrome nested problem page | 7/7 attention tests; explicit new affine and existing matrix probes passed |

The real-page check enters the solution through CodeMirror and runs the actual
local Pyodide/NumPy Worker. Added assertions exercise `addmm` forward/backward,
`addmv` and biased `functional.linear`, alongside mm/bmm/mv/dot/matmul and empty
dtype contracts. Static and numerical assets are real. Anonymous account/health
fixtures are supplied locally; external writes are blocked.

The complete application evidence verifier checks source, test, application,
policy, frontend, prior immutable and live export hashes, as well as stage log
hashes and the native mutation/reference reports. These wall times and absolute
budget passes are not speed comparisons with a different environment or run.

## Performance fixture and lifetime validation

The 124-flow performance fixture has passed output/gradient and dtype/shape
comparisons on CPU, Node/WASM and real Chrome/WASM. It covers 98 existing matrix,
SDPA and PaperLeet attention/GQA flows and 26 affine/Linear flows. All labels use
the same physical operand, tangent and gradient-seed arrays. Baseline A0 and A1
are identical 0.77 sources. New fused APIs compare against the equivalent
baseline composition, rather than a nonexistent older fused API.

Each runtime also passes 120 composition/fused lifetime observations: addmm and
addmv, float32 and float64, all eight input-gradient combinations, and retained
versus final backward. In every case the tracked nonleaf operand arrays and
summed output array are released after final backward while the scalar loss is
still alive. This establishes those logical array lifetimes; it does not prove
absence of every kind of leak or measure RSS, committed WASM heap or GPU memory.

CPython allocation tracing is separate and starts after fixture creation. For
the float32 32x64 by 64x48 addmm case with all three gradients active, the
composition retains 14,480 traced bytes after forward and peaks at 49,304 bytes;
the fused call retains 4,112 and peaks at 38,896. These exclude pre-existing
operand allocations and describe this one warmed fixture. The separate weak
references also show that fused backward does not need the nonleaf bias buffer.
No browser heap or model-wide memory reduction is inferred from these numbers.

## Ongoing work and evidence

Three fresh timing processes per CPU, Node/WASM and Chrome/WASM are now queued
sequentially. Each flow uses three batches of nine interleaved A0/A1/B rounds,
with at least 20 ms per recorded chunk. Every completed flow flushes its raw
rounds to the driver's log before the next flow. A/A drift and slower flows
require review; incomplete timing runs establish no broader speedup.

The original unexecuted performance preparation remains intact; the v2 driver
adds durable per-flow raw records. An idle pre-execution watcher was corrected
to run the additional Vite checks from the web directory; its prior source and
cancellation record are retained, and the live application job was untouched.
The historical partial checkpoint also has a verified snapshot: its observed
first four gate receipts are recovered byte-for-byte from the unchanged prefix
of the subsequently appended list. All original artifact hashes match in that
snapshot, and the original index bytes are preserved.

Evidence remains in the task's `torch-iteration-82` directory, including
`application-pre-latency-index.json`, the complete application verifier result,
the frozen eleven pre-latency stage receipts, three numerical fixture reports,
three lifetime reports, and the evolving timing logs. No timing result or
active-process receipt is represented as a completed performance qualification.

Remote main subsequently gained branding/header changes at
`aaa60e7152beb0aef1872d094fbb0454626111d8`; its 22 changed paths do not alter the
numerical library, application tests, problem sources or release policies.
Reconcile the then-current main and qualify its frontend/build/worker/page
inputs before adoption. Finish broader performance review and the verified
evidence archive, then use the existing fresh guarded main-push procedure.
Further core-API coverage and evidence-based backend selection remain part of
the independent-library goal.

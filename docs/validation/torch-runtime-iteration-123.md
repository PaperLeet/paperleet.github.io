# Torch runtime iterations 123–132: complete application and lifetime checks

The selected 0.79 runtime remains byte-identical to iteration 114, with 65
resources. Its implementation was committed locally as
`838ee29fd89176d3c3ac0d829b25fd1c9411e25a`. The current-main UI and description
changes were then merged at `33e147e3844b80408978f4cd3b7d4574a69189e3`, without
changing those runtime bytes. Main had been observed at
`4d032047aa9f5db71f3b4dfc5eac0db2e04855a9`; its library sources still matched
the 0.77 baseline.

## Application verification

The full CPU/application suite was run before and after reconciliation. Each
run passed 45,734 tests, with the same one conditional Docker integration test
skipped. The post-merge run froze 2,297 source, test, frontend and policy inputs.
The skip requires explicitly enabling the Docker harness after building its
image; this record does not claim that Docker scenario was executed.

The subsequent application gates ran serially:

| Gate | Result |
| --- | --- |
| Local native catalog, three repetitions | 89 modules, 12,360 test records; passed in 137.965 seconds against the existing 600-second budget |
| Judge export and source verification | 65 runtime resources matched standalone and private namespaces |
| Immutable assets | 15,687 existing assets preserved; 180 new assets reachable from active indexes |
| Active export identities | 183 payload/index hashes bound |
| Actual complete Pyodide/NumPy catalog | 89/89 modules, 406 mutants, concurrency 1; 489,553 ms; no browser budget exceeded |
| Full frontend checks and WASM driver | Passed |
| Pages release-policy checks | Passed; policies unchanged |
| Default and current Pages-profile builds | Both passed |
| Pages assets and production-asset retention | Passed |
| Generated Worker | Exact esbuild regeneration and immutable asset verification passed |
| Real Chrome nested problem route | Actual Worker executed the edited solution and all 7/7 tests passed |

The native command is the local `paperleet.cli ci` catalog runner. It did not
dispatch GitHub Actions. Its wall time is a budget observation, not evidence of
a library speedup against a previous run.

The nested-route test loaded the actual prerendered attention page, edited its
CodeMirror buffer, and ran local Pyodide/NumPy through the real judge Worker.
Alongside the matrix-entry-point checks, it exercised the new vector norm:
value 5, gradient `[0.6, 0.8]`, direct JVP 1.4, and HVP `[0.032, -0.024]` for
`[3, 4]` with an all-ones direction. The preexisting gradient buffer remained
unchanged after the functional transforms. Served runtime payloads contained
the 0.79 version. Only health/anonymous-account responses were fixtures; other
external requests were blocked, including writes.

All budget stages ran without other owned numerical, build, compression or
latency jobs. Their original child processes were awaited to terminal state;
observation timeouts did not cause restarts.

## Logical lifetime verification

Iteration 132 renewed the 120 affine lifetime cases on CPython/NumPy 2.5.4,
minimum NumPy 1.26.4, Node/Pyodide and Chrome/Pyodide. These cover all eight
operand-gradient combinations, retained backward and final release. Nonleaf
operand arrays were released after final backward while the scalar loss stayed
alive.

It also exercised 12 complete HVP flows, 25 calls each, on each of those four
runtimes. GC was disabled during each group. Weak references to every private
Tensor in the recording context were dead immediately after the function
returned, while the returned values remained alive. Each runtime completed all
300 calls with zero retained private Tensor owners. Reverse-phase state was
restored after each call.

These are logical ndarray and Tensor-owner observations. CPython tracemalloc
is reported separately. They do not measure process RSS, committed WASM heap or
GPU memory.

## Performance preparation and an observed optimization opportunity

The complete-version campaign now has 136 flows: 98 matrix/attention
regressions, 26 affine flows, and 12 HVP flows. The HVP set includes polynomial,
matrix, batched matrix, SDPA, and vector norms with orders 2 and 2.9. New norm
APIs compare with equivalent baseline compositions. Shared physical operands,
directions and gradient seeds and numerical agreement were checked on CPU,
Node and Chrome before timing.

Iteration 127 is the clean-source campaign snapshot. Each measurement block
contains all six Williams3 orders for baseline A0/A1 and candidate B, balancing
positions and ordered adjacent pairs within the block. Calibration and
measurement both disable GC. Calibration targets at least 100 ms chunks; the
recorded minimum must remain at least 40 ms. Three blocks form each flow, with
fresh processes for independent cycles. The stale caption in the preceding
preparation was corrected; its numerical Python AST was unchanged. No latency
measurement had been completed at this checkpoint.

An observational probe also found that fallback HVPs currently create recording
metadata they do not consume: 6 entries for the polynomial flow, 9 for matrix
flows, and 16 for SDPA, in each dtype. Norm flows consume their records. This is
an optimization candidate, not a demonstrated time or RSS regression. The new
campaign includes these complete flows so the cost can be assessed directly.
Earlier 0.78 timing anomalies and their controls remain retained; they are not
replaced by the new functional validation.

## Scope and evidence

The known exceptional numerical differences remain as described in the
[previous checkpoint](torch-runtime-iteration-111.md): 238 scalar-power point
differences, no new differing row against the actual 0.78 power baseline, and
one retained WASM norm/HVP point difference caused by final-power rounding
before sine. The original raw point gates remain failed. Passing application
checks does not turn those original comparisons into passes.

Raw inputs, results, durable logs and receipts are under
`/Users/david/Developer/paperleet-wasm-evaluation-20261015/torch-iteration-{123..132}`.
Iteration 129 is the post-merge full CPU proof, 131 covers the serial application
gates, and 132 covers logical lifetimes. The failed nested-path launcher in
iteration 128 is preserved; no test child launched there. Its corrected run
used the original root-level harness in iteration 129.

Overall performance qualification, a fresh final archive/audit, current upstream
revalidation and guarded main adoption remain pending. There was no main push,
CI dispatch, tag, publication or deployment. The broader core-API and WebGPU
objective remains active.

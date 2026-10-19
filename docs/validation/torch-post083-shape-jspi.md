# Post-0.83 shape reuse and implicit GPU suspension probes

Main adopted the qualified 0.83 runtime at
`7e4e8a500f3cb3a875964be08c03254d9c126d57`; its proof commit is
`8c0db0aa89c4bdae5bea17e75b4e4be0a78be952`. A fresh remote-tree check of the
proof commit verifies 66 runtime resources, 183 active export identities and
seven release-policy files. There are zero workflow runs for that commit and
Pages still uses `gh-pages` at `/`. Neither probe below changes production.

## Parent-shape reuse: no adoption

The isolated candidate resolves active graph parents before building binary
pullbacks and reuses their shape metadata. It aims to avoid allocating another
shape tuple for each selective derivative. All 93 existing SDK source files
are unchanged. The candidate passes 12,308 tests on NumPy 2.5.4, and 12,305 with
three existing skips on NumPy 1.26.4. All fields in an independent 882-case
arithmetic corpus are bitwise equal between baseline and candidate in each
NumPy environment.

One fresh NumPy 2.5 process measures all 24 existing owned whole-call flows,
including copying inputs and returned values/gradients. It uses separate A0/A1
baseline modules, balanced A0/A1/B orders, ordinary GC and the existing minimum
batch durations. All A/A, floor and 10% regression screen checks pass. This is
an exploratory screen, not a multi-process or multi-runtime qualification.

The target float32 scalar32 flow has candidate/baseline ratios of 1.00379
(ratio of medians) and 1.00279 (median of blocked ratios). That provides no
convincing speedup or evidence of recovering the previous 0.83 CPU cost. The
candidate is excluded; no favorable individual cases are rerun or selected.
Actual WASM qualification of this candidate is not claimed.

## JSPI: Python callbacks run during GPU upload suspension

A real Chrome 155 / Pyodide 0.29.4 / NumPy 2.2.5 probe loads the canonical 0.83
runtime into the existing opaque-origin classic Worker. It reuses the existing
bounded host GPU service, MatrixRpc client and explicit Matrix API with
unchanged default quotas, protocol, cancellation and disposal. The learner
Worker has origin `null` and no direct `navigator.gpu` access.

Each case queues an `asyncio.call_soon` Python callback, then synchronously
calls `Matrix.upload`, whose JSPI `run_sync` wait can suspend execution. The
callback records grad mode, evaluates an unrelated `x * 2`, and mutates the
input under a local `no_grad` context.

| Case | Callback stage | Callback grad mode | Unrelated result requires grad | Input version |
| --- | --- | --- | --- | --- |
| Ordinary live input | Inside upload | Enabled | Yes | 0 → 1 |
| Live input inside `no_grad` | Inside upload | Disabled | No | 0 → 1 |
| Owned snapshot before upload | Inside upload | Enabled | Yes | 0 → 1 |

All three uploaded matrices retain the original values; the existing RPC
sender copied bytes before the callback's mutation. Caller inputs contain the
later values. Grad mode is restored afterward, but an unrelated callback
observes the temporary global mode during the hidden wait. This observation
does not establish a native PyTorch asyncio parity difference.

Capturing a future autograd version guard only after GPU completion would see
version 1 and miss that uploaded bytes came from version 0. Capture versions
and any required owned values before the first suspension, then validate
before publishing graph/result state. An owned value snapshot alone does not
protect mutable Tensor identity or isolate grad mode. Define a serialized or
context-isolated execution contract before transparent GPU integration; avoid
holding process-global mode changes across hidden waits.

The probe sends ten requests, submits and completes six host operations and
peaks at 32 allocated bytes. Before closing, all handles and bytes are zero;
host disposal confirms zero allocations and a closed service. No automatic
Torch GPU storage or autograd backend is implemented, and no latency is
measured. A prepared predecessor failed compile preflight because module-level
`await` was incompatible with the fixture's `exec` entry. It was not executed;
the fresh probe uses explicit `run_sync` and retains that correction record.

[Machine-readable results](../../packages/stridetorch/benchmarks/results/post083-shape-jspi.json)
and the [verified raw archive](../../packages/stridetorch/benchmarks/results/post083-shape-jspi-harness.tar.gz)
preserve both candidates, all 93 SDK sources, native arithmetic records,
complete timing samples, the actual browser fixture and the proof-push receipt.
The archive contains 562 files, 223 unique payloads and SHA-256
`e93bf1cc6a9409d33ca69ddd61e2d96167e8bb89b4a91402164d5a8332454980`.
Replay requires the recorded native environments and the external Pyodide
0.29.4 runtime at stage 76, Chrome and Playwright. Adjust absolute paths when
replaying elsewhere. This evidence does not enable production WebGPU or change
the package name. The runtime improvement goal remains active.

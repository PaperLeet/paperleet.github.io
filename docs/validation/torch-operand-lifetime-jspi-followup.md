# Operand lifetime and experimental JSPI host transport

PaperLeet main has adopted the independently qualified
[0.82 runtime](torch-runtime-0.82-qualification.md). The experiments below keep
production sources unchanged. They are candidates for the continuing runtime
work, with allocation/latency and backend qualification still pending.

## Binary operand lifetime candidate

Ordinary multiplication and direct-division pullbacks need an operand's shape
without always needing its values. The candidate retains that shape and only
the arrays used by the active derivative. It preserves numerical helpers,
graph lifetime and the qualified 0.82 version guards. Both-active operations
continue retaining the values their derivatives read.

Two native NumPy environments pass 36 lifetime/analytic-gradient cases each.
An actual Chrome Worker passes the same 36 cases per variant. They cover
float32/float64, C/F layouts, scalar chains, tensor operands, mixed dtypes and
both-active controls. For a 64-by-64 float64 chain of 32 scalar multiplications,
the baseline captures 32 extra array owners totaling 1,048,576 bytes; the
candidate captures none. This measures captured operand owners before
backward. Peak allocation, RSS and latency remain unmeasured.

Both full native SDK runs pass: 12,298 on NumPy 2.5.4 and 12,295 with three
existing skips on NumPy 1.26.4. The unchanged browser SDK's 12,285 cases pass
in actual Node WASM and Chrome Worker contexts. The independent 478-case
clipping oracle also passes.

The initial Node invocation included an additional host test file: ten of its
cases passed and three subprocess cases failed because WASM cannot start a
native process. The original invocation remains failed. A fresh actual WASM
collection verifies all 12,298 node IDs, identifies the 13 host cases, and
reconciles the 12,285 unchanged browser cases without repeating the test run.
The first collection attempt's partial-output failure is also retained.

## Existing precision boundary

A new independent native Torch 2.6 corpus covers 882 arithmetic contexts per
native environment: broadcast shapes, C/F/sliced layouts, float16/32/64,
mixed dtypes, aliases, scalar operands and gradient state. Baseline and
candidate outputs, losses, gradients and state match byte for byte throughout.
Native comparisons match all assertions in 876 contexts; six mixed-precision
contexts exceed the original strict float64-gradient tolerance.

Those six differences are already present in the 0.82 baseline. Additional
controls locate different float32 cotangents upstream of the binary pullback.
Supplying the identical native cotangent makes every compared binary gradient
match byte for byte. The original comparison remains failed and its tolerance
is unchanged. The diagnostic provides a concrete precision boundary for future
work rather than a claim of complete native parity. An earlier alias test
setup error is preserved separately.

## Experimental opaque Worker to host GPU transport

On Chrome 155 with Pyodide 0.29.4, the opaque Worker has origin `null`, an
insecure context and no exposed GPU API. A classic Worker matching PaperLeet's
current worker type nevertheless uses `run_sync` from an asynchronous Python
entry to await a MessageChannel request serviced by the trusted host GPU.
The host runs a fixed 2-by-2 matrix kernel and returns the expected owned result
`[19, 22, 43, 50]`. The synchronous-entry control rejects stack switching;
an unregistered operation is rejected. All four created buffers are destroyed,
and the observed request ledger has no pending requests or late replies.

Two preceding module-Worker attempts fail before runtime loading and remain
archived. This observation establishes a transport option. Tensor input
ownership, cancellation, device loss, timeout/late-reply handling, gradient
state, CPU fallback and complete-call performance still need qualification.
Production WebGPU remains disabled. Pyodide documents this stack-switching
API as [experimental](https://pyodide.org/en/0.29.4/usage/api/python-api/ffi.html#pyodide.ffi.run_sync).

[Machine-readable results](../../packages/torchwasm/benchmarks/results/operand-lifetime-jspi-followup.json)
and the [verified raw archive](../../packages/torchwasm/benchmarks/results/operand-lifetime-jspi-followup-harness.tar.gz)
retain frozen variants, all test sources, native reference data, full results,
failed invocations and replay dependencies. The archive contains 559 files,
253 unique payloads, with SHA-256
`b317bd96d6a2bebae9603c2dad0d5d86ca897bd05fff89085ad3c056651477ce`.

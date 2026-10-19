# Experimental WebGPU matrix service

This developer prototype keeps the Python execution Worker inside an opaque-origin
`sandbox="allow-scripts"` iframe. A separate secure host Worker owns the GPU device.
The sandbox receives a MessagePort capability for a fixed matrix protocol. These
files do not enable a torch GPU device or change the supported CPU/WASM runtime.

The three implementation files have no PaperLeet imports. `gpu-service.mjs` runs
as a module Worker, or exports `createGpuEngine` and `servePort` for a trusted host.
`rpc-client.mjs` implements the MessagePort client. `matrix_client.py` provides
explicit resident float32 2D matrices in Pyodide, with upload, multiplication,
transpose flags, owned readback and context-manager release.

The trusted application creates the host Worker and MessageChannel, transfers one
port to the service with `{type: 'connect'}`, and transfers the other port to its
Python Worker. In the Python Worker, construct `new MatrixRpc(port)`, await its
`ready` promise and set `self.matrix_rpc = client` before loading the Python
module. Python must be entered with `runPythonAsync` or another JSPI-compatible
entrypoint. Host code must own Worker shutdown and an independent deadline.
Channel delivery and origin/source/nonce checks belong to the trusted application.
The archived fixture demonstrates one such handshake; it is not a production
judge integration.

```python
with Matrix.upload(a) as left, Matrix.upload(b) as right:
    with left.matmul(right) as output:
        result = output.numpy()
```

Default limits are 32 MiB of tracked GPU buffers, 4 MiB per matrix, dimensions up
to 1024, 268,435,456 multiply-accumulate steps per multiplication, 24 resident
handles, eight pending operations, 4096 requests and a 120-second session lease.
Readback staging and uniform buffers count toward the same allocation limit.
Clients cannot upload WGSL, change these limits or obtain the device object.
Cancellation drops queued work or discards a completed result; it does not
preempt an already submitted GPU kernel. Closing destroys owned buffers/device
and invalidates the session. Zero counters do not measure physical VRAM reclaim.
MessagePort queues and browser/driver resources do not have a hard memory or
denial-of-service isolation guarantee from these API limits.

Run the portable deterministic protocol checks with a recent Node.js:

```sh
node packages/stridetorch/benchmarks/webgpu_rpc/check-protocol.mjs
```

These checks inject a fake engine. Real GPU/Pyodide numerical, lifecycle and
latency evidence is in [iteration 74](../../../../docs/validation/torch-runtime-iteration-74.md).
The raw harness includes the browser fixture, native Torch oracle generator,
36 fresh-process measurements and an earlier MIME configuration failure. It
retains historical absolute paths and requires Playwright, Chrome, native Torch
and local Pyodide assets to reproduce. The 2D primitives and explicit VJPs do not
constitute automatic torch autograd, batched/broadcast matmul, attention,
float64/complex/special-value coverage or a GPU training backend.

## Bounded multi-operation programs

`MatrixProgram` groups up to eight input descriptors, eight fixed matmul nodes
and eight distinct computed outputs into one request. Add all inputs before
nodes. Inputs can be float32 NumPy arrays (copied when `run` submits them) or
live `Matrix` objects owned by the same client/session. References returned by
`input`/`matmul` identify values inside this program; forward/cyclic references
and arbitrary shader operations are rejected.

```python
program = MatrixProgram()
ar, br, gr = [program.input(value) for value in (a, b, gradient)]
y = program.matmul(ar, br)
da = program.matmul(gr, br, transpose_b=True)
db = program.matmul(ar, gr, transpose_a=True)
with program.run([y, da, db]) as result:
    output, a_gradient, b_gradient = result.arrays
```

These are explicit analytical matrix derivatives, rather than automatic torch
autograd. `run(..., read=False, retain=True)` returns resident outputs in
`result.matrices`; use them as inputs to later programs. Request both flags to
obtain independent CPU snapshots and resident matrices. The context manager
closes all still-live retained outputs in one validated `release_many` call;
individual Matrix objects also retain their ordinary `close` method.

The host validates the entire DAG, shapes and peak byte/work/handle requirements
before allocating. The 268,435,456 multiply-accumulate limit applies to the
**sum** of all nodes. The 32 MiB allocation limit includes new uploads, computed
values, uniforms and the packed readback buffer, plus existing resident buffers.
Large programs may need partitioning even when each individual product is valid.
No default budget is increased by this interface.

One command submission contains the compute passes and final readback copies.
Only requested retained outputs survive the transaction. Ordinary rollback
drains already queued writes before releasing tracked buffers; failure to drain
invalidates the device and its existing handles. Cancellation still skips queued
requests or discards all newly retained outputs after execution. It does not
preempt submitted GPU work.

Portable failure-injection and transaction checks:

```sh
node packages/stridetorch/benchmarks/webgpu_rpc/check-program.mjs
```

See [iteration 75](../../../../docs/validation/torch-runtime-iteration-75.md) for
three program ownership modes, native numerical controls, real multi-output
quota/cancellation checks and sequential-versus-program measurements. The initial
short-batch measurements are retained. The longer measurement host raises only
its trusted request counter quota to support longer sampling; the default
4096-request limit, memory/work/handle limits and production runtime are unchanged.


### Primitive failure and cancellation recovery

Upload, matmul and readback failures now wait for possibly issued queue work
before destroying temporary buffers or reclaiming their tracked byte quota.
A failed drain closes the device and invalidates all resident handles. An
acknowledged cancellation takes precedence over a later execution error
while the session remains open; the execution queue advances after rollback.
Normal program transactions keep the same planner, shader and default limits.

Portable checks share the injected surface in `gpu-fixture.mjs`:

```sh
node packages/stridetorch/benchmarks/webgpu_rpc/check-primitive.mjs
node packages/stridetorch/benchmarks/webgpu_rpc/check-cancel-drain.mjs
node packages/stridetorch/benchmarks/webgpu_rpc/check-program.mjs
node packages/stridetorch/benchmarks/webgpu_rpc/check-protocol.mjs
```

The first two cover 39 primitive lifecycle cases and four acknowledged
cancellation/failure races. Set `GPU_PRIMITIVE_RESULT` or
`GPU_CANCEL_DRAIN_RESULT` to save JSON results. Real Chrome GPU queue,
numerical-recovery and device-destruction evidence is recorded in
[iteration 78](../../../../docs/validation/torch-runtime-iteration-78.md).
This remains an explicit experiment and does not add a Torch GPU device.
## Matched Torch forward/VJP workloads

`torch_workload.py` supplies a NumPy-only fixture generator and two matched
providers. `TorchMatrixStep` runs the actual CPU library's `a @ b` and
`autograd.grad(output, (a, b), g)`; `GpuMatrixStep` builds explicit GPU forward,
left-VJP and right-VJP programs. Both return three independent CPU snapshots.
The GPU provider is an experiment, not a Torch storage or autograd backend.

Cases are `(37,53,29)`, 64³, 256³ and 512³, with cold and resident inputs.
Resident setup uploads three matrices once; calls still read every result back.
Cold mode includes all uploads. The 512³ workload uses two programs to respect
the unchanged aggregate MAC limit, including repeated cold input uploads.

The current real Chrome controls match native Torch 2.6/2.14 values, retain
read-only inputs and release handles/allocations. These controls contain no
latency measurement. Earlier raw NumPy CPU baselines do not establish a speedup
over the improved Torch CPU path. See the
[0.80 qualification record](../../../../docs/validation/torch-runtime-0.80-qualification.md)
before drawing any GPU routing or performance conclusion.

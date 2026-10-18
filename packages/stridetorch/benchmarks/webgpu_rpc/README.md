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

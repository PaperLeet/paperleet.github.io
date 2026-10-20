# Torch runtime iteration 74: isolated WebGPU matrix service

The supported library remains **0.73.0**, executing on CPU/NumPy/WASM. This
iteration adds a reusable developer prototype and measured evidence. It does not
enable a torch GPU device, change import aliases or alter the production judge.

An opaque-origin `sandbox="allow-scripts"` iframe retains its isolation. Its
Pyodide Worker has no direct WebGPU access. A MessagePort connects it to a secure
host Worker that owns one device and exposes only fixed float32 2D matrix
operations. An 8×8 tiled WGSL kernel accepts transpose flags; resident results
can feed subsequent matrix products without a CPU read between them.
The Python client copies upload/readback data and binds every handle to its
own session. The portable sources have no PaperLeet imports:
[`benchmarks/webgpu_rpc`](../../packages/torchwasm/benchmarks/webgpu_rpc/README.md).

## Verification

* 36 native Torch 2.6.0 and independent float64 matrix/analytical-VJP fixtures,
  nine irregular/tile-boundary shapes and all four transpose combinations.
  Original tolerance: `rtol=1e-4`, `atol=5e-4`; no tolerance was widened.
* 13 real opaque-worker boundary checks, including shape/field/handle errors,
  residency/transient/work quotas and recovery. A separate real cancellation
  check verifies that an in-flight result is discarded.
* Nine deterministic protocol/lifetime checks with an injected fake engine.
* Four additional real GPU lifecycle checks: pending-startup disposal,
  resident-handle invalidation, abandoned-port lease recovery and a real device
  deliberately destroyed to exercise the device-lost event. These additional
  clients run in the trusted fixture page; they do not extend opaque-worker or
  hardware-crash coverage.
* 66 namespace/feature-detection tests and 9 built-wheel/ownership/independent-install tests.
  The stable runtime's 61 source resources and exported runtime identities
  remain identical. Full application CPU/catalog/frontend tests from iteration
  73 were not rerun for these new benchmark-only files.

The browser was unmodified headless Chrome 155.0.8059.82 on Apple Metal3, without
extra feature flags. The fixture supplied CORS for local Pyodide 0.29.4 assets.
It verifies parent-document access remains blocked, authenticates frame messages
by source and nonce, and terminates the learner Worker before reporting success.
Every successful device session ended with zero tracked allocations/handles.
This checks API accounting and destruction calls, not immediate physical VRAM
reclaim. An earlier local server MIME failure and its exact inputs are retained.

## Whole matrix-program costs

Each of six programs ran in six fresh Chrome/Pyodide/device processes. CPU A and
A′ are identical NumPy/WASM programs. GPU roundtrip includes per-call upload,
Python/JS conversion, RPC, dispatch, readback and release. Resident keeps inputs
on the GPU between calls; its setup upload is excluded, while output readback
and release remain included. Adapter/pipeline/Python initialization is excluded
from these warm measurements. Every program first checks float32 results
against an independent float64 expression, caller arrays and resource balance.

Eight balanced batches per program/profile target 20ms. Request-budget caps and
timing variation leave **111/1152 batches** below target, with minimum CPU/A′
19.6/19.5ms and GPU roundtrip/resident 10.1/6.2ms. These samples remain visible;
the results support the observed large differences, not a precise universal
crossover. Ratios are medians of six per-process ratios; descriptive microsecond
medians are aggregated separately and need not reproduce those ratios.

| Program (M×K×N) | NumPy μs | GPU roundtrip μs | GPU resident μs | A′/A | Roundtrip/A | Resident/A |
| --- | --- | --- | --- | --- | --- | --- |
| paperleet_attention_score_head (4×5×6) | 0.91 | 1757.81 | 968.75 | 0.9965 | 2055.1640 | 1128.2203 |
| attention_score_128 (128×64×128) | 479.30 | 1948.44 | 1179.69 | 0.9910 | 4.0093 | 2.5065 |
| dense_256 (256×256×256) | 11425.00 | 3156.25 | 1840.63 | 0.9977 | 0.2654 | 0.1637 |
| dense_512 (512×512×512) | 97800.00 | 9468.75 | 6662.50 | 0.9974 | 0.0944 | 0.0661 |
| resident_mlp_chain (128×64×32) | 143.75 | 2654.69 | 1703.12 | 1.0069 | 18.9603 | 11.9932 |
| linear_vjp (128×64×32) | 348.44 | 4287.50 | 3256.25 | 0.9891 | 11.9569 | 9.3380 |

`paperleet_attention_score_head` is one 4×5 by 5×6 score product from the
attention fixture, not full attention. `resident_mlp_chain` is two products with
no activation. `linear_vjp` explicitly evaluates the output and two analytical
matrix derivatives on both sides; it does not compare automatic torch autograd.

On this measured setup the 256/512 dense products benefit from the GPU even with
per-call transfers, while the small score matrices, short chain and small VJP
are slower. Residency reduces costs but does not make these small programs
faster. Keep CPU/WASM as the supported default. Further GPU work should retain
data across larger programs, amortize dispatch, qualify the Tensor/storage and
autograd interfaces, and test the real judge's async execution/budgets before
selecting an explicit opt-in backend. These results do not select an automatic
routing threshold or establish native CPU, other-browser, full-model or training
performance.

## Boundaries and reproduction

The service bounds accepted GPU allocations, handles, pending requests, work
and lease duration. It does not provide arbitrary shader execution. Cancellation
skips queued work or discards an output after completion; device destruction
does not establish GPU kernel preemption. MessagePort transport queues and
browser/driver resources still lack a hard denial-of-service isolation guarantee.
FP64/complex/special-value behavior, batched/broadcast/empty products and a
public GPU/autograd backend remain unqualified.

The [Pyodide 0.29.4 JSPI documentation](https://pyodide.org/en/0.29.4/usage/api/python-api/ffi.html#pyodide.ffi.run_sync)
requires an asynchronous-compatible Python entry for `run_sync` and labels the
feature experimental. The [WebGPU device destruction documentation](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/destroy)
distinguishes device/resource destruction from already queued command execution.

Machine report: `packages/stridetorch/benchmarks/results/iteration-74.json`.
Raw harness `iteration-74-harness.tar.xz`: 1,725,688 bytes,
SHA-256 `6b36ce6007548a48ce6ad93e40ef6300bd275e458eeda8ac1e24a61ce00ee8ae`; 139 logical files,
120 mode/ownership-compatible payloads and
19 hardlinks, every logical extracted payload verified.
The raw harness preserves absolute historical paths and requires Chrome,
Playwright, Pyodide assets and native Torch setup/path adjustment.
The qualification base is `ecc3938b7d2268b944136e829df04e549b5ffee3`; remote integration is verified
separately. No CI, tag, package publication or deployment is requested here.

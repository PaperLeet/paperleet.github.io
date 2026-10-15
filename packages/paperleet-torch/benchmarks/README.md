# Runtime benchmarks

These are developer tools, not CI timing assertions. Use the same machine,
runtime, sources, dtype, workload and sampling configuration for each comparison.
Do not overlap timing runs with tests or other benchmarks. Results retain every
sample, the median, runtime versions and imported implementation path.

From the repository root, with the standalone package installed:

```sh
python packages/paperleet-torch/benchmarks/bench_runtime.py --repeat 11 --iterations 30 --output cpu.json
node packages/paperleet-torch/benchmarks/bench_wasm.mjs --output wasm.json
```

The WASM runner uses the repository's pinned Pyodide installation and package
cache. Override `--runtime` and `--cache` when needed. `--source` selects the
package source directory, allowing a frozen baseline to run in a separate fresh
process. CPU baselines can use `PYTHONPATH=/path/to/baseline-parent` with the
same benchmark file. Inspect the recorded implementation path to verify imports.

Use `--dtype float64` to measure double-precision workloads. Repeat `--case`
to select a smaller set, for example `--case attention_backward --case linear_backward`.
`bench_attention_memory.py` separately records forward retained/peak and backward
peak allocations with CPython tracemalloc; it must not be treated as browser RSS.

Each case warms up three calls and reports milliseconds per call over repeated
batches. Forward/backward cases reset gradients each call; fixture construction,
imports and serialization are excluded. `adam_step` measures updates with a fixed
gradient. Matmul-only cases use frozen inputs. These measure kernel/graph work,
not cold start, training convergence or complete application latency.

`model_construct` includes layer initialization and container registration for a
16→32→8 MLP. `model_training_step` includes gradient reset, the same MLP's forward,
backward and SGD update on a 16-row batch. `model_parameter_walk` enumerates that
model's parameters. These common workloads also run against the 0.2 baseline;
new checkpoint features are checked for correctness separately.

Optimizer workloads use separate parameters: `adam_step`, `adamw_step` and
`sgd_step` update fixed gradients on a 256×256 parameter; `adamw_zero_grad` clears
an existing gradient, `adam_construct` constructs an optimizer over eight existing
64×64 parameter fixtures, and `same_dtype_to` measures one no-op dtype conversion.
Very small browser samples are sensitive to timer quantization.

`cpu_noop` and `contiguous_noop` measure already-CPU/already-C-contiguous Tensor
calls. `contiguous_copy` packs a transposed input. Conversion/packing cases include
their Tensor/graph allocations when a copy is needed. A no-op is expected to
preserve identity in the current implementation; benchmark historical versions
with the same script to expose their unnecessary allocations.

`bench_schedulers.py --output schedulers.json` measures one/eight-group step
schedules, cosine (float and scalar Tensor rates), warmup/cosine, plateau, and
complete plain/scheduled MLP updates. The scheduler-only cases time `step()` after
one setup optimizer update, without parameter updates or gradients in the timed
calls. Plain/scheduled MLPs use identical seeded initial weights and inputs.
Pass `--backend native` for an optional native Torch CPU comparison; the default
imports only this library and NumPy and can run unchanged inside Pyodide. Scheduler
construction is excluded from these timings. All samples and runtime identities
are retained, and no timing assertions run in CI.
Use `node packages/paperleet-torch/benchmarks/bench_wasm.mjs --benchmark schedulers
--iterations 120 --output schedulers-wasm.json` for the same suite on pinned WASM.

`bench_nn.py` measures analytical LayerNorm/RMSNorm (all gradients and frozen-input
cases), frequency-scaled embedding, exact/tanh GELU, dropout and complete encoder
training. Each backward case clears relevant gradients. The 8×16-token encoder
uses width 32, four attention heads, a 32→64→32 feed-forward block with tanh GELU
and dropout, two normalized/residual paths, cross entropy and SGD momentum.
Its constructor/fixtures are excluded; the step includes zeroing, forward,
backward and update. New-module cases are omitted from historical runtimes that
do not expose those classes; six common functional cases remain comparable.
Use `bench_wasm.mjs --benchmark nn --dtype float32 --iterations 120` for Node WASM.

`python packages/paperleet-torch/benchmarks/bench_optimizer_memory.py --dtype float32`
records CPython tracemalloc allocations after creating Adam over 16 pre-existing
256×256 parameters and after one parameter receives a gradient. It also records
state-array bytes. This is allocation evidence for lazy state, not RSS or browser
heap usage; parameter fixture storage is excluded.

For browser measurements, serve these files and the pinned runtime from a local
HTTP origin. Load `bench_runtime.py` and package sources into a fresh Pyodide
Worker, import `bench_runtime`, and call `run_benchmarks(repeat=11, iterations=30)`.
Compare versions sequentially in fresh Workers. Benchmark JSON includes the
runtime and workload details needed to distinguish Node from browser results.

`webgpu_matmul.mjs` exports `benchmarkMatmul()` for a secure-context browser:

```js
const {benchmarkMatmul} = await import('./webgpu_matmul.mjs');
console.log(await benchmarkMatmul());
```

It compiles a tiled float32 WGSL kernel, checks every output element against an
independent JavaScript sum (including an irregular shape), and records compile,
upload/compute/readback and resident-dispatch timings separately. The resident
case repeats a matmul with the same resident inputs; it does not model a neural
network computation chain. GPU validation errors fail the run, and all buffers
and the device are released. No adapter is a reported unsupported result.

The GPU probe is an experiment, not a torch device implementation. Its roundtrip
time excludes Python/JS conversion and autograd and is therefore a lower bound
for integrating it with this library. Compare it to matmul-only WASM results in
the same browser, not to a forward/backward benchmark or to native BLAS. Browser
timer quantization, adapter/driver variation and batching affect the results.

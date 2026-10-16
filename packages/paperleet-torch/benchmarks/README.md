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

`bench_cpu.py --benchmark runtime --threads 1 --output cpu.json` sets OMP/MKL/
OpenBLAS/vecLib limits before NumPy imports and records the actual NumPy BLAS
provider. On supported Apple Accelerate builds it also selects the single-thread
BLAS/LAPACK policy and checks it before/after the run. Without that API, it reports
environment configuration without claiming a verified thread count. Other
arguments pass through to the selected runner. The direct bench_*.py scripts
leave thread setup to their caller. CPU and WASM results record source hashes.

`bench_tensor_factories.py` separates builtin float/int/bool construction, a small
Python list, a NumPy scalar, 64×128 array copies/imports/casts, transposed sharing,
existing-Tensor copies and identity-preserving `as_tensor`. Each timing includes
the complete allocation/import and temporary Tensor cleanup; fixture construction
is outside timing. `--dtype` controls array fixtures, while builtin scalar defaults
remain float32/int64/bool. CPU/WASM runners accept `--benchmark tensor_factories`;
use 1,200 calls per batch. WASM also copies and hashes available LICENSE/NOTICE
resources separately from Python source hashes.
Small scalar/list/identity cases use ten times that inner count to reduce browser
clock quantization; every result records its actual iteration count.
`numpy_export` and `numpy_array_protocol` separately measure shared-data exports
and use 100 times the inner count for their sub-microsecond metadata operations.

The 0.12 construction suite has 24 cases, adding small integer/boolean lists,
64/4,096-element Python vectors, explicit floating/integer storage and nested
primitive rows. Every new case checks values/dtype outside timing, and reports
shape/dtype contracts for comparison across versions. Large 4,096-element cases
use `max(1, iterations // 40)` calls per batch. Source/destination fixtures are
prepared before timing; complete Tensor allocation and cleanup remain included.

`bench_arithmetic.py` separates same-dtype Tensor/Tensor and Tensor/Number
add/sub/mul/div, reflected scalar subtraction/division, complete add/mul/div
reverse passes and a 16-stage scalar affine chain. Fixed 64×128 float32/float64
values, result dtypes and analytical gradients are checked outside timing.
Use `--benchmark arithmetic` with the CPU/WASM runners: defaults are eleven
batches, 1,200 calls for forward cases, 120 for short reverse cases and 30 for
the chain, after three warmups. Intermediate allocation and cleanup remain
included. Ordinary fixture timing does not substitute for separate extreme-value,
signed-zero, dtype/Number and callback-policy correctness contracts.

`bench_selection.py` measures same-dtype `where`, float-scalar `where`, `cat`,
`stack`, their complete reverse passes and a 16-stage where chain on 64×128
float32/float64 fixtures. Values, dtypes and analytical gradients are checked
outside timing; masks select two of every three positions. Use `--benchmark
selection` with the CPU/WASM runners. The default is 11 batches of 120 calls,
with 30 calls for the chain. A float scalar ensures equal baseline/candidate
semantics; changed mixed-dtype correctness is tested separately from latency.

`bench_where_memory.py` retains a squared-mean scalar loss through a 16-stage
where chain after dropping branch/intermediate user variables. Input and mask
storage are allocated before tracing. Default/retained backward and final release
record traced retained/peak allocations, with independent values/gradients
checked after tracing stops. These are CPython allocation measurements, not RSS
or browser heap usage. Use `--dtype float64` for the wider fixture.

`bench_tensor_hooks.py` measures plain leaf/graph backward, retained non-leaf
backward, leaf observers/transforms/post-accumulation callbacks, intermediate
registration/dispatch and explicit registration/removal. Each case uses a
separate fixture; ordinary leaf hooks are registered outside timings. The
intermediate cases include registration because each forward creates a new
intermediate Tensor. Graph cases reset gradients and evaluate a squared mean
through a square on 16×32 inputs. Leaf cases reverse fixed seeds on 16 elements.
Contiguous/strided copy cases use frozen 64×128 inputs.
Registration/removal uses ten times the requested inner count. Historical
runtimes without Tensor hooks still execute three common reverse/retention cases.
Use `bench_wasm.mjs --benchmark tensor_hooks --iterations 1200` on pinned WASM.

`bench_module_hooks.py` measures unhooked/removed-hook/active-hook dispatch,
16-layer identity composition, frozen linear inference, module construction,
checkpoint saving/loading and registration/removal. Run it on CPU, or pass
`--benchmark module_hooks` to the WASM runner. The default has eleven batches of
1,200 calls; small dispatch/construction/registration cases use ten times that
count to reduce browser timer quantization. Each result records its actual inner
iteration count. Hook-free baselines execute the six common cases; callback
cases require a version that implements their registration methods. Cases are
separate fixtures, and callback registration is outside timings except in the
explicit `register_remove` case.

Use `--dtype float64` to measure double-precision workloads. Repeat `--case`
to select a smaller set, for example `--case attention_backward --case linear_backward`.
`bench_attention_memory.py` separately records forward retained/peak and backward
peak allocations with CPython tracemalloc; it must not be treated as browser RSS.
`bench_graph_memory.py` keeps a scalar loss alive through backward after dropping
intermediate user variables, comparing default release and explicit retention.
It traces a 64×128 input through sixteen add/tanh stages and a squared mean loss;
input allocation precedes tracing. Use `--dtype float64` for the wider fixture.

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

`bench_batchnorm.py` compares the joint NumPy BN kernel with a first-order Tensor
composition in this same library. It validates values/gradients before timing
tiny/MLP/vision training backward, eval backward, frozen-input affine/bias cases
and no-grad training forward. Select `--implementation joint` or `composite`;
the reference is not the native saved-statistic JVP tape and is not native Torch
or a previous unsupported BN API. Imports/fixtures are excluded, each backward
clears gradients, and all process/batch samples must be retained. Use
`bench_wasm.mjs --benchmark batchnorm --implementation composite --dtype float64`
for the same reference on pinned WASM. The iteration 16 record includes eval
backward costs as well as the training gains; preserve dynamic running-buffer
semantics when considering further optimizations.

The GPU probe is an experiment, not a torch device implementation. Its roundtrip
time excludes Python/JS conversion and autograd and is therefore a lower bound
for integrating it with this library. Compare it to matmul-only WASM results in
the same browser, not to a forward/backward benchmark or to native BLAS. Browser
timer quantization, adapter/driver variation and batching affect the results.

GroupNorm's first-order joint/composition benchmark uses the same library for
both implementations and validates values and active gradients before timing:

```sh
PYTHONPATH=packages/paperleet-torch/src python packages/paperleet-torch/benchmarks/bench_groupnorm.py --implementation joint --dtype float32
node packages/paperleet-torch/benchmarks/bench_wasm.mjs --benchmark groupnorm --implementation joint --dtype float32
```

Use separate fresh processes in balanced order; preserve all samples. Results
measure these local kernels, not native Torch or application-wide acceleration.

Classification kernels include independent first-order NumPy value/VJP checks:

```sh
PYTHONPATH=packages/paperleet-torch/src python packages/paperleet-torch/benchmarks/bench_classification.py --implementation joint --dtype float32
node packages/paperleet-torch/benchmarks/bench_wasm.mjs --benchmark classification --implementation joint --dtype float32
```

Label same-source joint/composition comparisons separately from old-release
comparisons. The 0.17 baseline supports only `tiny_index`, `classifier_index` and
`vocabulary_index`, using `--implementation joint`. Preserve all process samples,
freeze source/harness identities and exclude concurrent correctness/build work.

`bench_convolution.py` measures complete Tensor convolution forward and squared-mean
backward calls, including validation, packing, graph construction and cleanup.
Ten cases cover tiny/pointwise/spatial/depthwise/dilated/3D/larger inputs and
forward-only, weight-only and bias-only tapes. `--implementation auto|direct|matrix`
selects a private experiment override; it is not a public convolution argument.
Every active gradient and output is checked against the same-library direct
path before timing; independent native/scalar tests establish mathematical
correctness separately. Use `bench_wasm.mjs --benchmark convolution` for pinned
WASM. Default sampling is nine batches, twenty calls and three warmups; tiny
cases multiply calls by four and the larger case divides by eight. All timing
samples are retained. A direct/auto comparison is neither native-Torch latency
nor whole-application speedup.

# Torch runtime iteration 1: indexing, attention and Adam

This is an iteration toward an independent library, not a claim of complete
PyTorch compatibility or a stable release. The active repository is
`PaperLeet-public`; its default upstream branch is `main`, not `master`. The
working tree also contains unrelated ongoing application work. No mixed release
commit or deployment is part of this iteration.

## Implementation

Basic integer/slice/ellipsis/inserted-axis indexing uses a unique-destination
gradient update. Advanced and boolean indexing retain unbuffered scatter-add,
including duplicate indices. Native-reference cases cover non-contiguous
storage, empty selections, shared paths and graph reuse with different seeds.

The independent raw-NumPy attention oracle separates its score and probability
pullbacks. Query, key and additive mask now share one softmax pullback per
backward traversal instead of recomputing it up to three times. No persistent
seed/gradient cache is used. Repeated backward calls, frozen operands, masks,
empty sequences and saved-storage version guards remain covered.

Default attention scaling previously used a NumPy float64 scalar, promoting
float32 inputs and outputs under NumPy 2.x. The factor is now a Python scalar so
attention preserves its input dtype. Native-reference dtype assertions and a
real-WASM float32 regression cover this separately from numerical closeness.

Adam now counts updates separately for each parameter. A parameter's first
gradient uses its first-step bias correction even if other parameters have
already trained. Coupled weight decay is supported, malformed gradient
shape/dtype is rejected, and `zero_grad(set_to_none=False)` preserves absent
gradients and existing gradient-buffer identity. Native comparisons include
late-starting, intermittently absent and permanently absent gradients. Parameter
groups and optimizer checkpoint/state APIs remain future work.

## Benchmark methodology

The benchmark source and execution instructions live in
`packages/paperleet-torch/benchmarks/`. Measurements use three warmup calls,
11 batches and 30 iterations per batch, with identical seeded float32 inputs.
Imports and fixture construction are excluded. Timings are recorded as samples
and medians; they are not CI speed assertions. Versions run sequentially in
fresh processes or browser Workers, without concurrent tests started by this
task during timing runs.

The baseline is the frozen pre-iteration standalone package source. CPU results
record its imported file path to detect accidental editable-package reuse.
Real Node and Chrome WASM measurements use Pyodide 0.29.4 / NumPy 2.2.5. Native
CPython uses Python 3.12.11 / NumPy 2.5.4 on this Apple M4 machine. Browser
measurements use headless Chrome 155.0.8059.40.

Attention speedups combine shared-gradient work and the corrected float32
arithmetic; the baseline incorrectly promoted that workload to float64. They do
not prove that every API or dtype is faster. Linear backward, duplicate-index
scatter, Adam and matmul-only cases are controls, not optimized implementations
in this iteration. An alternate Adam denominator expression measured slower and
was removed while keeping its correctness fixes.

## Memory tradeoff

CPython tracemalloc includes NumPy allocations tracked by that runtime; it is not
a browser RSS measurement. With the 2×4×64×32 attention fixture and additive mask,
float32 forward peak decreased from 1,062,358 to 474,262 bytes, while retained
forward allocations decreased from 399,510 to 337,046 bytes.

For float64, the shared graph retains an additional logits array: retained
allocations increased from 398,192 to 664,344 bytes and backward peak increased
from 1,391,568 to 2,047,976 bytes. Remove that saved intermediate through a shared
multi-input VJP before calling this memory-optimized attention. This limitation
is recorded rather than hidden behind the float32 speedup.

## WebGPU feasibility boundary

An independent tiled float32 WGSL matmul probe ran on an Apple `metal-3` adapter.
It checks every result element against an independent JavaScript dot-product
calculation, including a 37×53 by 53×29 tail shape. Compilation, full
upload/dispatch/readback and resident-dispatch timings are recorded separately;
the resident case repeats one operation and is not a neural-network chain.

The probe does not implement a torch GPU device, storage model, transfers,
autograd or Python/JS bridging. Its roundtrip measurements are a lower bound on
full backend cost. Large browser matmuls justify prototyping a resident backend;
small matrices do not justify automatically routing every operation to the GPU.
Complete chain timings and explicit unsupported-operation behavior are required
before PaperLeet adopts such a backend.

The host integration also requires design work: the current production Worker
enters Python through synchronous `runPython`, while WebGPU buffer mapping is
asynchronous. Pyodide 0.29.4's experimental `run_sync` requires JS Promise
integration and a compatible async Python entry point. A synchronous alias alone
does not solve GPU synchronization. See the [buffer mapping API](https://developer.mozilla.org/en-US/docs/Web/API/GPUBuffer/mapAsync)
and the [pinned Pyodide API](https://pyodide.org/en/0.29.4/usage/api/python-api/ffi.html#pyodide.ffi.run_sync).

Adam's update/weight-decay and missing-gradient contracts were checked against
the [native optimizer documentation](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html)
and independently executed native PyTorch tests. No external small project's
implementation was adopted.

## Validation and remaining scope

- Final `python -m pytest tests packages/paperleet-torch/tests -q`: 1,449 passed,
  one optional Docker test skipped and one existing native PyTorch deprecation
  warning. This run includes the final dtype fix, 42 new native kernel/optimizer
  cases and all configured original reference/mutant suites.
- The focused native/core run passed 217 cases after the dtype fix.
- The built independent wheel passed all 10 standalone tests under isolated
  `python -I` outside the checkout with NumPy 1.26.4 and no native PyTorch.
  An additional float32 attention forward/backward smoke check also passed in
  that minimum-dependency environment. Wheel sources match the current source
  tree and NumPy remains its sole runtime dependency.
- All five representative real Pyodide suites passed after re-exporting final
  sources. The maintained check now includes dtype preservation, slicing versus
  duplicate indexing, delayed Adam updates and finite-difference SDPA gradcheck.
- Fresh Chrome Workers passed the final exported Schedules, Attention, LoRA,
  AdamW and Muon Split bundles: 35 original tests and all 20 mutants caught.
- The exported-source bundle check passed for all 69 modules. No frontend UI
  behavior or production GPU backend was introduced in this iteration.

The continuing scope, next core APIs, backend requirements and stable integration
gates remain in `packages/paperleet-torch/ROADMAP.md`. Float64 saved memory,
model/optimizer lifecycle coverage and a resident WebGPU tensor/graph prototype
remain open; this iteration does not complete the overall library goal.

## Measured results

| Runtime / workload | Before (ms) | After (ms) | Speedup |
| --- | ---: | ---: | ---: |
| CPython / slice_backward | 0.1800 | 0.0538 | 3.34× |
| CPython / attention_backward | 0.4409 | 0.2298 | 1.92× |
| CPython / adam_step | 0.0754 | 0.0753 | 1.00× |
| Node WASM / slice_backward | 0.3740 | 0.1347 | 2.78× |
| Node WASM / attention_backward | 3.5587 | 2.7384 | 1.30× |
| Node WASM / adam_step | 0.2880 | 0.2997 | 0.96× |
| Chrome WASM / slice_backward | 0.4000 | 0.1333 | 3.00× |
| Chrome WASM / attention_backward | 3.5867 | 2.7133 | 1.32× |
| Chrome WASM / adam_step | 0.2933 | 0.2933 | 1.00× |

These are workload-specific measurements. Control cases stay approximately flat;
the data does not establish an across-the-library speedup.

| Matmul M×K×N | Chrome WASM (ms) | GPU roundtrip (ms) | GPU resident (ms) |
| --- | ---: | ---: | ---: |
| 32×32×32 | 0.0167 | 0.4000 | 0.0500 |
| 64×128×128 | 0.4567 | 0.5000 | 0.0800 |
| 128×128×128 | 0.9033 | 0.4000 | 0.0700 |
| 256×256×256 | 10.2133 | 1.0000 | 0.4300 |
| 512×512×512 | 90.8933 | 2.2000 | 0.7600 |
| 37×53×29 | 0.0267 | 0.3000 | 0.0400 |

GPU compilation and each raw sample are preserved separately in the evidence.
The Python/JS bridge and a resident multi-operation graph are still unimplemented.

Raw samples, environments, source hashes, dtype and memory measurements are
preserved in [`iteration-1.json`](../../packages/stridetorch/benchmarks/results/iteration-1.json).
The complete baseline source snapshot and local runtime logs remain in the
`paperleet-wasm-evaluation-20261015/torch-iteration-1` evidence folder.

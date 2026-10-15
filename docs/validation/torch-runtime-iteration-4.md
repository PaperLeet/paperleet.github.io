# Lightweight torch 0.4 validation

This milestone adds a shared optimizer lifecycle and complete in-memory training
checkpoint workflows. SGD, Adam and AdamW remain independent raw NumPy oracles
for PaperLeet's learner implementations. The standalone distribution still has
only NumPy as a runtime dependency; native Torch is used for reference tests.

## Behavior

- `torch.optim.Optimizer` provides parameter groups, defaults, optional named
  parameter inputs, `add_param_group`, gradient zeroing and checkpoint packing/
  loading. Per-group learning rates and algorithm options are read on each step.
  Constructor defaults and actual groups are validated; unknown execution options
  are not silently ignored. Native duplicate-group errors and warned-about
  duplicates within a group are covered, including SGD's first momentum step.
- Adam moments and per-parameter step counters are created only on the first
  gradient. SGD has lazy momentum buffers. AMSGrad, maximization, coupled Adam
  decay, decoupled AdamW decay, delayed gradients and changing learning rates
  match native fixtures. Frozen/no-gradient parameters do not acquire state or
  receive weight decay.
- State dictionaries preserve native positional index mappings, optional names,
  shallow tensor references, global metadata and group order. Loading casts
  moments to current parameter dtypes, preserves parameter identity and handles
  legacy numeric Adam counters. Names do not determine remapping. Same-dtype
  loaded tensors remain shared with the supplied checkpoint; snapshots require
  `deepcopy`, as with native Torch. Wrong group counts/sizes fail before replacing
  the current groups; state shapes/types are checked before numerical updates.
- `zero_grad(set_to_none=False)` preserves existing gradient objects/storage and
  absent gradients. Step closures enable gradient recording, return their result
  and restore the outer mode even after exceptions. Parameter/state updates keep
  storage-version guards, so saved graphs reject mutated operands.
- Supported aliases include `torch.optim.optimizer`, `.sgd`, `.adam` and `.adamw`.
  Ordinary scripts can keep those imports under the existing launcher/host
  integration. Legacy single-group `lr` and `params` access remains available.
- Same-dtype `Tensor.to` returns the original Tensor instead of allocating a graph
  node. This also preserves identity through JVP and inside gradient-mode contexts.
  Actual dtype changes retain their existing reverse/forward derivative rules.

CPU float32/float64 parameters and frozen scalar Tensor hyperparameters are
supported. Hooks, enabled foreach/fused/capture/differentiable execution,
trainable hyperparameters, zeroing non-leaf gradient graphs, file serialization
and a GPU execution backend remain unsupported. This is in-memory library state,
not native `.pt` file compatibility or a complete Torch implementation.

## Checks

- After integrating unrelated current-main UI changes, the complete Python suite
  passed: **1,567 tests**, one optional Docker test skipped, one existing upstream
  Torch deprecation warning. The focused final core run passed **447 tests**.
- New native-reference optimizer cases cover grouped updates, delayed state,
  dynamic addition, model/optimizer save-rebuild-load-resume, closures, zeroed
  buffer identity, dtype casts, shallow state aliases, invalid groups/options,
  metadata, duplicate parameters, scalar/empty/non-contiguous parameters, gradient
  storage aliases, live Tensor learning rates and positional named remapping.
  These run against installed native PyTorch 2.14.1.
- Native lesson gate: **69 modules, three repetitions, 8,829 test executions,
  all 295 mutants caught**, 113.46 seconds in this run.
- Actual Pyodide **0.29.4 / NumPy 2.2.5**: five full reference/mutant/learner suites
  (schedules, attention, LoRA, AdamW, Muon split), existing multifile and numerical
  fallback checks, six model contracts and **12 optimizer workflow contracts**.
  The maintained WASM checker runs both independent contract files on exports.
- Chrome **155.0.8059.40**, five fresh Workers: **35 original tests, all 20 mutants,
  18 additional model/optimizer contracts**, package version 0.4.0 and actual
  `/judge/paperleet_torch` imports confirmed.
- Independent wheel installed outside the checkout, isolated Python and **NumPy
  1.26.4 with no native Torch**: **36 standalone tests passed**. The wheel is
  64,625 bytes; its hash is in the raw record. Both application-only and standalone
  wheel/export/launcher tests pass, retaining disjoint installed paths.
- Rebased full web checks, production build and the real WASM checker passed.
  Numerical source hashes remained unchanged through the UI-only rebase. All
  tracked published assets are retained; only this worktree's untracked and
  unreachable intermediate exports were removed.

An earlier full run still had a live process when the UI rebase began. Its result
was not used as final integration evidence; the fresh post-rebase full run above
completed successfully. No correctness jobs were running during timed benchmarks.
The repository license remains undecided and no PyPI publication is performed.

## Measurements

Baseline: frozen `dcbd26b` / library 0.3.0. Candidate: frozen 0.4.0 sources, with
per-file hashes checked after rebase. Each version runs sequentially in a fresh
process or Worker, three warmups then **11 batches × 120 calls**, identical seeded
fixtures and separate parameter fixtures for each optimizer workload. CPU uses
Python 3.12.11 / NumPy 2.5.4, OMP/MKL/OpenBLAS environment variables set to 1. Node v25.9.0 and Chrome
use the pinned WASM runtime above. Imports/fixture creation are excluded except
where construction itself is the named workload.
The small MLP is 16→32→8 on a 16-row batch, including gradient reset, forward,
backward and SGD update in each timed step.

CPU configuration clarification: these variables record environment settings;
they do not establish Apple Accelerate's threading policy. Historical samples
remain unchanged. The [0.10 record](torch-runtime-iteration-10.md) describes the
later BLAS provider audit and directly verified Accelerate controls.

| float32 workload | CPU 0.3 → 0.4 (ms) | Chrome WASM 0.3 → 0.4 (ms) |
| --- | ---: | ---: |
| Adam step, 256×256 | 0.07476 → 0.07609 | 0.28833 → 0.27583 |
| SGD momentum step, 256×256 | 0.02963 → 0.02620 | 0.10167 → 0.10417 |
| AdamW step, 256×256 | 0.13379 → 0.08053 | 0.36750 → 0.29667 |
| AdamW gradient zeroing | 0.00345 → 0.00247 | 0.02000 → 0.01750 |
| Adam construction, eight 64×64 parameters | 0.01249 → 0.00401 | 0.03833 → 0.01000 |
| Small MLP training step | 0.06451 → 0.06780 | 0.12833 → 0.13583 |

Avoiding eager Adam arrays reduces construction time. Reusing gradient buffers
avoids replacement allocations. Scalar Adam bias correction uses Python math,
keeping float32 intermediates in float32 under NumPy 2.x. Float32 AdamW is about
40% faster on CPU and 19% faster in Chrome for this fixture. This is not an
across-library speedup: float64 AdamW CPU/Chrome updates are approximately flat,
and the small MLP training step costs about 2–6% more across measured runtimes
with the additional lifecycle/validation work. Repeated scalar parsing was
removed after profiling the initial implementation's larger overhead.

Same-dtype conversion measures about 0.00127 → 0.00020 ms on CPU float32 and
0.00362 → 0.00065 ms in Node WASM. The operation no longer allocates a Tensor/VJP
node. Chrome's very small conversion samples are timer-quantized, so an exact
browser speedup is not claimed. Attention and linear controls remain near their
baselines. The Node float64 AdamW baseline differs from Chrome's; raw samples are
retained and no consistent float64 improvement is inferred from that run.

### Lazy-state allocations

The memory fixture pre-creates **16 parameters of shape 256×256** and then starts
CPython tracemalloc. It measures registered NumPy allocations and Python objects,
not RSS, browser heap or total process memory. Only one parameter receives a
gradient in the measured step.

| Allocations / float32 | 0.3 | 0.4 |
| --- | ---: | ---: |
| Constructor moment-array bytes | 8,388,608 | 0 |
| Constructor total retained bytes | 8,394,032 | 2,344 |
| State-array bytes after one gradient | 8,388,608 | 524,292 |
| Total retained after one gradient | 8,660,228 | 796,284 |
| Peak registered allocation bytes | 9,446,984 | 1,321,056 |

For float64 the constructor's 16,777,216 moment bytes also become zero; after one
parameter's gradient the new state has 1,048,580 bytes, including its scalar
counter. Fully active training still requires the ordinary moment arrays; this
saving is for lazy construction and inactive parameters, not a claim that Adam
no longer needs moments.

Raw samples, environments, source identities, memory and validation summaries are
in [iteration-4.json](../../packages/paperleet-torch/benchmarks/results/iteration-4.json).
Use [the benchmark tools](../../packages/paperleet-torch/benchmarks/README.md),
including `bench_optimizer_memory.py`. Local complete logs, frozen baselines,
prototype measurements, fresh-Worker scripts and the sequential runner are under
`paperleet-wasm-evaluation-20261015/torch-iteration-4`.

Model dtype/device transforms, schedulers/hooks, broader tensor/nn coverage,
allocation profiling and resident WebGPU storage/graph execution remain in the
[roadmap](../../packages/paperleet-torch/ROADMAP.md). The independent-library goal
remains active beyond this repository integration milestone.

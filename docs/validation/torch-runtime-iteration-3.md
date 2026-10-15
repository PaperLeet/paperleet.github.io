# Lightweight torch 0.3 validation

This milestone adds registered model lifecycle, containers and in-memory
checkpoints to the independent NumPy-backed library. Ordinary `import torch`,
`from torch import nn` and `torch.nn.functional` imports continue through the
existing host aliases or `paperleet-torch script.py` launcher. There is no native
PyTorch dependency in the standalone wheel and no production GPU backend.

## Behavior and boundaries

- `Module` tracks parameters, child modules and persistent/nonpersistent buffers.
  Named traversal matches native order, supports prefixes and duplicate control,
  and deduplicates shared layers/parameters by default. Registration, replacement,
  deletion, `None` entries and child lookup are validated.
- `train`, `eval`, `apply`, `requires_grad_` and `zero_grad` preserve model/parameter
  identity. Recursive mode/application/state calls respect child overrides.
- `Sequential`, `ModuleList`, `ModuleDict`, `ParameterList` and `ParameterDict`
  compose models; tests cover slices, mutation, order and shared references.
  ParameterList rejects a Tensor passed as its iterable, matching native behavior.
- `state_dict` includes persistent buffers, alias paths, metadata and subclass
  extra state. It returns shallow references unless copied. `load_state_dict`
  copies into existing storage, preserves optimizer references, casts into the
  destination dtype and reports missing/unexpected keys. Wrong shapes/non-Tensor
  entries fail even with `strict=False`; valid earlier entries can be copied
  before a later error, matching native loading behavior.
- Deep-copied checkpoint tensor views share a copied allocation, independent of
  the source. Copied model Parameters start without gradients; ordinary leaf
  Tensor copies retain independent gradient copies. Saved-graph mutation guards
  still reject backward through parameters changed by checkpoint restoration.
- Invalid registration names now raise native-style `KeyError`. Earlier tests
  expecting unsupported state dictionaries or `ValueError` for those names were
  updated to validate the new behavior against native PyTorch.

This remains a bounded implementation. Hooks, assign-based loading, model
`to`/dtype transforms, optimizer checkpoint state, `torch.save/load` and general
higher-order autodiff remain unsupported. ModuleList insertion requires indices
from zero through its length; named Sequential insertion fails explicitly.
Cyclic recursive state/mode operations, non-leaf tensor copies and exotic
non-dense NumPy backing allocations also fail explicitly. These limits are not
inferred from merely passing the lesson catalog.

## Verification

- Full Python infrastructure, native-reference and standalone suite: **1,496
  passed**, one optional Docker test skipped, one upstream Torch deprecation
  warning. Native-reference model tests run against installed PyTorch 2.14.1.
- Final focused model, alias, tensor, buffer, wheel and application checks:
  **303 passed**. The independent model suite includes unchanged-import training,
  snapshot/restore and continued updates through the existing optimizer.
- Native reference/mutation/determinism gate: **69 modules, three repetitions,
  8,829 test executions, all 295 mutants caught**, 108.74 seconds in this run.
- Actual Pyodide 0.29.4 / NumPy 2.2.5: five full reference/mutant/learner suites
  (schedules, attention, LoRA, AdamW and Muon split), original multiple-test-file
  regression, existing numerical/typed-fallback checks, and six independent
  model/checkpoint contracts against the final exported library.
- Chrome 155.0.8059.40, fresh Workers for those same five bundles: **35 original
  tests, all 20 mutants, six additional lifecycle tests**. Workers reported
  package version 0.3.0 and the actual `/judge/paperleet_torch` module location.
- Built 0.3.0 wheel, installed outside the checkout with **NumPy 1.26.4 and no
  native Torch**: **24 standalone tests passed** under isolated Python. Wheel
  size is 60,331 bytes. Application and standalone distributions retain disjoint
  installed paths; built-wheel application-only export and launcher tests pass.
- Full frontend checks and production build passed. The final export is used by
  the browser checks. All tracked historical assets are retained; only this
  worktree's untracked, unreachable intermediate exports were removed.

The repository still has no license file; the independent public package license
is undecided. This milestone does not publish a package to PyPI.

## Performance

The baseline is frozen remote main `bc663da` (library 0.2.0), and the candidate is
frozen 0.3.0 source. Each version runs sequentially in a fresh process/Worker,
with three warmups and 11 batches of 120 calls. Both dtypes use identical seeded
fixtures. CPU uses Python 3.12.11 / NumPy 2.5.4 and one OMP/MKL/OpenBLAS thread;
Node and Chrome use the pinned Pyodide/NumPy runtime above. No correctness jobs
were run concurrently with these timing measurements.

The small MLP is 16→32→8 on a 16-row batch. Construction includes initialization,
allocation and registration; a training step includes gradient reset, forward,
backward and SGD. Measurements below are milliseconds per call.

| Runtime / dtype | Construct 0.2 | Construct 0.3 | Train 0.2 | Train 0.3 |
| --- | ---: | ---: | ---: | ---: |
| CPU / float32 | 0.02947 | 0.03562 | 0.06753 | 0.06536 |
| CPU / float64 | 0.03666 | 0.04393 | 0.06560 | 0.06610 |
| Node WASM / float32 | 0.07141 | 0.09355 | 0.12625 | 0.12681 |
| Node WASM / float64 | 0.10368 | 0.10995 | 0.12846 | 0.12539 |
| Chrome WASM / float32 | 0.07667 | 0.09500 | 0.13250 | 0.12917 |
| Chrome WASM / float64 | 0.09083 | 0.11000 | 0.12833 | 0.13417 |

Complete model registration adds construction cost: roughly 6–7 microseconds
on CPU and 18–19 microseconds in Chrome for this fixture. Chrome training-step
medians change by −2.5% / +4.5%; these samples do not establish a broad speedup.
Linear, attention, basic-slice and indexed-backward controls vary within about
7% across these runs, in both directions, with unchanged numerical algorithms.
Parameter-walk timings are very small and sensitive to browser timer granularity;
no throughput improvement is claimed from them.

Raw samples, benchmark identity, source hashes, wheel hash and gate summaries are
in [iteration-3.json](../../packages/paperleet-torch/benchmarks/results/iteration-3.json).
Use the existing [benchmark tools](../../packages/paperleet-torch/benchmarks/README.md)
with `--case model_construct`, `--case model_training_step` and
`--case model_parameter_walk` to reproduce the added workloads. Complete logs,
frozen sources, Chrome Workers and the sequential runner are retained locally in
`paperleet-wasm-evaluation-20261015/torch-iteration-3`.

Optimizer parameter groups/state/resume, model conversion, additional tensor and
neural-network APIs, allocation profiling and a resident WebGPU execution chain
remain in the [roadmap](../../packages/paperleet-torch/ROADMAP.md). This is a main
integration milestone; the independent-library objective remains active.

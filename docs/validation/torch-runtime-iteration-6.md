# Lightweight torch 0.6 validation

This milestone completes common grouped learning-rate scheduling and scheduled
training continuation. The runtime still depends only on NumPy; existing code can
import `torch.optim.lr_scheduler` through the host alias or script launcher.

## Behavior

- Supported schedules: StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR,
  LinearLR, ConstantLR, PolynomialLR, LambdaLR, MultiplicativeLR, SequentialLR,
  ChainedScheduler and ReduceLROnPlateau. Custom LRScheduler subclasses can return
  iterable rates. Native initialization applies epoch zero immediately and
  establishes initial/base rates and step counters.
- Current-rate schedules follow manual rate changes and supported dynamic group
  additions; base-rate/lambda/floor metadata retains native constraints. Duplicate
  milestones compound decay, zero milestones affect initialization, and sequential
  transitions reset the active child's epoch. Cosine continues beyond its first
  period. Chained schedules apply each child in order.
- Plateau scheduling supports min/max, relative/absolute thresholds, patience,
  cooldown, per-group minima, epsilon, NaN/infinite metrics and continuation state.
  Scalar minima expand for added groups; explicitly supplied minima must be
  adjusted by the caller. Metric-based schedules cannot be composite children.
- State dictionaries retain counters, base/last rates, child state and callable
  object attributes while retaining the destination optimizer. Lambda functions
  are reconstructed by the caller; executable code is not serialized. Load
  scheduler state before optimizer state after constructing both objects; scheduler
  loading alone does not restore optimizer group rates. Snapshots need deepcopy.
- Frozen scalar Tensor learning rates retain their group object/storage identity;
  values are filled in place with version bumps. Initial/base/reported tensors
  are copies. Multiple groups sharing the same rate follow native read-then-fill
  behavior. Updating such a rate invalidates older graphs that saved it.
- Optimizer-step tracking is installed only when a scheduler is constructed and
  uses a weak reference, avoiding retention of discarded optimizers. First-step
  ordering/override warnings, rate inspection warnings and deprecated explicit
  epoch stepping are covered. Step return values/closures remain unchanged.

Cyclic/one-cycle/warm-restart policies, differentiable scheduling hyperparameters,
general metric-based composition and file/function serialization are not provided.
Forward-AD rate inputs fail explicitly rather than silently losing tangents.
Some nested ChainedScheduler→SequentialLR boundaries have no native closed-form
transition and raise NotImplementedError; the covered cases preserve that error.
The production runtime remains CPU/NumPy. The separate JSPI/WebGPU experiments
are not an implemented Torch GPU storage or gradient backend.

## Efficiency

Profiling the first implementation found redundant Tensor-rate copies and generic
autograd construction for frozen scalar math. The final path copies snapshots
directly, reads original rates before arithmetic, and uses typed NumPy scalar
coefficients for eligible float32/float64 rate calculations. Each operation keeps
the rate dtype on both NumPy 1.x and 2.x; generic Tensor behavior remains available
for other cases. The final group update still fills the original Tensor and
bumps its normal storage version.

Baseline controls use frozen `202d247` / 0.5.0 sources; candidate uses frozen 0.6.0
sources. Source hashes are retained and all **26 Python sources** match after
rebasing unrelated UI changes through `fbde557`. Every timing run is sequential
in fresh processes/Workers with no local correctness jobs running. CPU is Python
3.12.11 / NumPy 2.5.4, OMP/MKL/OpenBLAS environment variables set to 1; Node v25.9.0 and Chrome
155.0.8059.40 use Pyodide 0.29.4 / NumPy 2.2.5. Workloads use three warmups and
11 batches × 120 calls. Construction/imports are excluded from scheduler timings.

CPU configuration clarification: these variables record environment settings;
they do not establish Apple Accelerate's threading policy. Historical samples
remain unchanged. The [0.10 record](torch-runtime-iteration-10.md) describes the
later BLAS provider audit and directly verified Accelerate controls.

| Scheduler workload (ms) | CPU | Node WASM | Chrome WASM |
| --- | ---: | ---: | ---: |
| Step, one group | 0.000476 | 0.001381 | 0.001667 |
| Step, eight groups | 0.001404 | 0.003413 | 0.003333 |
| Cosine, eight float rates | 0.001848 | 0.004365 | 0.004167 |
| Cosine, eight Tensor rates, initial | 0.088960 | 0.233175 | 0.233333 |
| Cosine, eight Tensor rates, final | 0.031338 | 0.076739 | 0.073333 |
| Warmup→cosine, eight groups | 0.002050 | 0.004366 | 0.004167 |
| Plateau, eight groups | 0.000648 | 0.001339 | 0.001667 |
| Plain MLP training | 0.068082 | 0.130065 | 0.126667 |
| Scheduled MLP training | 0.069278 | 0.126229 | 0.130833 |

The targeted Tensor-rate path improves approximately **65% CPU, 67% Node WASM and
69% Chrome** versus the initial implementation, not versus a previously released
scheduler. Native Torch 2.14.1 CPU Tensor-rate cosine measures 0.028611 ms in the
same fixture; this library's final CPU path is still about 10% slower there.
This comparison is not an across-library performance claim.

The MLP is 16→32→8 on a 16-row batch, including zeroing, forward, backward and SGD;
the scheduled version adds StepLR. Both use identical seeded initial weights.
CPU adds about 1.2 microseconds and Chrome about 4.2 microseconds in this fixture.
Node's reversed difference is within measurement variability and is not a
scheduling speedup. Very small browser samples are timer-quantized. Plateau timing
uses repeated identical metrics and eventually reaches its minimum rates.

Common operator controls include attention, linear, slicing, advanced indexing,
optimizer updates and an unscheduled MLP. Node/Chrome controls stay approximately
flat. One initial CPU float32 candidate run was anomalously slow (some array
workloads 72–149% higher) while float64 was stable. The cause is not established.
The original results are retained. A separate balanced **baseline→candidate→
candidate→baseline** rerun used 11 × 360 calls per case in fresh processes, with
the same frozen sources and environment settings. The anomaly did not recur;
both versions measured comparable latency. No CPU kernel speedup is inferred.
Adam lazy-state allocation controls retain their previous zero constructor
moment bytes; scheduling adds no per-parameter moments.

## Validation

- Full Python suite at `ef72ecd`: **1,904 passed**, one optional Docker test
  skipped, five warnings originating in native Torch, 152.07 seconds.
- **191 native-reference scheduler cases** cover rates and actual optimizer
  updates, both dtypes, scalar Tensor rates, manual changes, checkpoints, custom
  subclasses, callable attributes, dynamic groups, nested boundaries, warnings,
  invalid arguments, shared rates and saved-graph guards. Forty-two cases compare
  float32 output bits exactly at ordinary, tiny and subnormal rates.
- **Nine new NumPy-only contracts** cover grouped SGD against independent NumPy,
  warmup/cosine AdamW save-rebuild-load-resume, plateau continuation, Tensor-rate
  identity/snapshots, standard aliases, warnings, shared storage mutation guards
  and explicit unsupported forward differentiation.
- Native lesson gate: **69 modules, three repetitions, 8,829 executions and all
  295 mutants caught**, 133.06 seconds in this run.
- Real pinned WASM: five complete reference/mutant/learner suites (schedules,
  attention, LoRA, AdamW and Muon split), existing numerical/multifile/typed-fallback
  regressions, and **37 independent lifecycle contracts**. The maintained checker
  now includes scheduler_lifecycle alongside the model/optimizer/conversion files.
- Chrome: five fresh Workers, **35 original tests, all 20 mutants caught and
  37 additional contracts**. Version 0.6.0 and actual `/judge/paperleet_torch`
  imports are asserted. Web checks and production build pass. A final UI-only
  rebase through `fbde557` retains all Python/export source hashes; affected
  current-main web checks and the production build are rerun after that rebase.
- Installed independent wheel outside the checkout, isolated Python with **NumPy
  1.26.4 and no native Torch: 55 tests passed**. Wheel **77,180 bytes**, SHA-256
  `f4e11286922d32fb722852e1da7934abb613636b4c3090e06f21deaa70b14f42`.
  Application/private-vendor and standalone packaging/export/launcher checks
  pass as part of the full suite, preserving disjoint installed paths.
- Supplementary independent native golden data: **42 trajectories / 504 float32
  outputs** also match bit-for-bit in the installed NumPy 1.26.4 wheel and in a
  separate Chrome/Pyodide Worker. These cross-runtime probes supplement the
  maintained native tests; their scripts and complete fixtures remain in the
  local evidence directory, rather than being claimed as additional CI cases.

Raw samples, environments, source hashes, initial/final scheduler paths, balanced
CPU controls, memory and validation metadata are in
[iteration-6.json](../../packages/paperleet-torch/benchmarks/results/iteration-6.json).
Use [bench_schedulers.py](../../packages/paperleet-torch/benchmarks/bench_schedulers.py)
and the WASM runner's `--benchmark schedulers` option. Complete logs, frozen
sources and fresh-Worker scripts are in the local
`paperleet-wasm-evaluation-20261015/torch-iteration-6` evidence directory.

Only untracked, unreachable intermediate exports in this owned worktree are
eligible for cleanup; tracked published assets remain available. No producer was
live during the UI rebase. No license is selected and no PyPI release is made.
The broader core/efficiency/independent-library and resident GPU goals remain
active in the [roadmap](../../packages/paperleet-torch/ROADMAP.md).

# Lightweight torch runtime development

The goal is an independently usable torch-compatible library with efficient
computation, broader core API coverage and validated CPython/browser behavior.
PaperLeet remains the integration workload, not the definition of all supported
behavior. Passing its current exercises alone does not establish library readiness.

## Version 0.2 milestone

- Preserve ordinary `torch` imports through the host alias or script launcher.
- Speed up basic slicing backward while retaining duplicate-index accumulation.
- Share the analytical attention pullback through a multi-input graph node, with
  no retained logits array or cross-backward gradient/seed cache.
- Preserve float32 attention outputs under NumPy 2.x instead of promoting them
  through a NumPy float64 scaling scalar.
- Fix Adam's per-parameter update counts, coupled weight decay and missing-gradient
  zeroing. Validate against independent native PyTorch, not another adapter path.
- Maintain reproducible CPython/WASM benchmarks and a separately labeled WebGPU
  feasibility probe with full upload/dispatch/readback timings and value checks.

## Version 0.3 milestone

- Registered model lifecycle: recursive train/eval/apply, shared-module and
  parameter/buffer traversal, four registered containers and Sequential slicing.
- In-memory state dictionaries with metadata, extra state, persistent buffers,
  strict key validation, native copying rules and retained optimizer references.
- Independent checkpoint copies preserve tensor storage views; copied model
  Parameters start without gradients. Native-reference and NumPy-only tests cover
  training, checkpoint restoration and continued updates under CPython and WASM.
- At this milestone, hooks, assign-based loading, model device/dtype conversion,
  file serialization and optimizer checkpoints were outside the contract.

## Version 0.4 milestone

- Common optimizer lifecycle for SGD/Adam/AdamW: parameter groups, named inputs,
  dynamic group addition, mutable learning rates, closures, in-memory snapshots
  and positional checkpoint remapping into current parameter objects.
- Lazy Adam moments and per-parameter counters, AMSGrad/maximization, native
  missing-gradient rules, reusable zeroed gradients and state mutation guards.
- Same-dtype Tensor conversion preserves identity, including during JVP, avoiding
  needless graph nodes. Raw NumPy updates remain independent learner oracles.
- Native-reference/standalone train-save-rebuild-load-resume validation and
  construction/update/zeroing/cast/memory benchmarks; unsupported execution modes
  and file serialization remain explicit boundaries.

## Version 0.5 milestone

- Recursive CPU model dtype/device conversion preserves Parameter and existing
  gradient identities, optimizer references and integer buffer rules.
- Tensor conversion overloads, memory-format/copy behavior, CPU device metadata,
  and explicit backend availability queries support unchanged CPU-selection code.
- Gradient modes support nested contexts, direct setters and recursive/generator
  decorators, with caller-mode restoration on suspension, return and exceptions.
- Linked storage-version counters retain mutation guards across model storage
  rebinding. Saved-self, saved-view and saved-output backward cases are compared
  with native Torch, including underflow and mixed-dtype errors. Old live graphs
  are operator-specific and are not an unrestricted compatibility promise.
- Allocation/hot-loop measurements check the cost of richer graph metadata and
  conversion semantics. CPU and real WASM tests include independent conversion,
  continued-training and in-memory checkpoint contracts.

## Version 0.6 milestone

- Twelve common learning-rate schedules share grouped initialization, last-rate
  inspection, optimizer-step tracking and in-memory continuation state.
- Linear warmup, cosine/step/polynomial decay, lambda/callable policies, chained
  and sequential composition and metric-driven plateau scheduling run through
  ordinary `torch.optim.lr_scheduler` imports.
- Native-reference tests cover rate trajectories, actual updates, Tensor-rate
  identity, manual rate changes, custom subclasses, dynamic groups, nested
  boundaries and save-rebuild-load-resume behavior. Independent NumPy-only
  training/checkpoint workflows are checked in real WASM.
- Scalar Tensor rates remain frozen CPU values. Cyclic/one-cycle/warm-restart
  policies, general metric-driven composition and executable function/file
  serialization are separate work.

## Next implementation areas

1. Remaining model lifecycle: common hooks where bounded behavior can be
   validated; define file checkpoint serialization and expand memory formats.
2. Optimizer extensions: remaining scheduling policies, common hooks and algorithms where
   numerical and performance evidence justify them; keep state compatibility and
   unknown-option errors explicit. Profile small-parameter update overhead.
3. Core tensor/autograd coverage: audit shape/dtype promotion, views and mutation,
   multi-output gradients, retention semantics and forward-mode gaps. Add APIs
   with native value, VJP/JVP and boundary tests, independent of course coverage.
4. Neural-network coverage: expand common normalization/module APIs, attention
   options and losses, with masked/empty/non-contiguous cases and numerical checks.
5. Runtime efficiency: profile allocation/graph overhead and larger linear algebra
   workloads. Track import size, cold start, steady-state latency and memory as
   separate measurements. Retain baselines and record regressions, not just wins.
   The multi-input VJP removes the prior float64 logits-retention regression;
   larger attention sequences still need memory and throughput measurements.
6. Optional WebGPU backend: keep storage resident across operations, provide
   explicit device/transfer semantics, feature detection and unsupported-operation
   policy, and validate analytical gradients. Establish an end-to-end benefit
   before enabling it for PaperLeet. A WGSL matmul probe is not an implemented
   torch GPU backend; Python/JS transfer and graph execution remain required work.

## Stable integration and independent release gates

- The default upstream branch is currently `main` (`origin/HEAD`), corresponding
  to the user's request to adopt mature versions on the main/master branch.
  Prepare a focused, reproducible integration that excludes unrelated concurrent
  UI/backend edits; do not treat an uncommitted working tree as a released version.
- Both independent and private vendored wheel layouts work outside the checkout
  and have disjoint installed paths. No dependency on the application, catalog,
  native PyTorch or browser assets is allowed in the independent runtime.
- Maintain native-reference core API tests and a standalone test entry point;
  NumPy is the only runtime dependency. Check packaging metadata, license,
  documented scope, installation, CLI and import-alias behavior before publishing.
  The repository currently has no license file; the independent distribution's
  license must be decided before a public package release.
- Run the complete application infrastructure suite, all configured original
  reference/mutant suites, exported-source checks and representative real WASM
  regressions. New backend behavior also needs real browser Worker evidence.
- Document supported API/dtype/gradient/device boundaries and performance under
  identified environments. Preserve typed unsupported errors and server fallback
  behavior; never silently pass off another backend's result as GPU execution.
- A stable repository version is an integration milestone. Independent-library
  readiness and optional GPU adoption remain separate goals requiring evidence.

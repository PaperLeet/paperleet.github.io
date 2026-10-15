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

## Next implementation areas

1. Model lifecycle: recursive train/eval, named parameter/buffer traversal, shared
   modules, containers and in-memory state save/load with native alias/copy rules.
2. Optimizer lifecycle: parameter groups, lazy per-parameter state, state save/load,
   checkpoint resume and learning-rate updates. Unify duplicated update machinery
   where parity and benchmarks support it; preserve unknown-option errors.
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

# Lightweight torch 0.5 validation

This milestone adds recursive CPU model conversion and gradient-mode contexts,
setters and decorators. The independent package continues to use NumPy only;
PaperLeet bundles identical sources in its private namespace and browser assets.
Ordinary `import torch` remains available through the host alias or launcher.

## Behavior and boundaries

- `Module.to`, `.cpu`, `.float`, `.double` and `.type` recurse through registered
  children and honor child `_apply` overrides. Parameter objects and existing
  gradient objects retain their identity, so existing optimizer references work.
  Integer/bool buffers are retained by floating conversions; `.type` casts them.
  Shared parameters remain shared. Converted buffer entries follow native
  replacement rules, which can split aliases. State snapshots keep old storage.
- Tensor conversions accept device/dtype/other-Tensor overloads, `tensor=`,
  `copy`, `non_blocking` and supported memory formats. CPU-index conversions
  force copies while plain CPU/same-dtype conversions preserve identity.
  `contiguous()` avoids copying C-contiguous inputs; actual copies retain VJP/JVP
  behavior. `clone()` preserves dense layout by default. Same-dtype `.to` can
  retain a transposed layout even with `contiguous_format`, as native Torch does.
- Device metadata has native-style type/index, representation and equality.
  Tensors execute on logical CPU only. CUDA availability/initialization queries
  return false and the device count is zero; CUDA/MPS/meta execution still fails
  explicitly. Device indices are bounded to 0–127. Channels-last memory formats,
  differentiable complex conversions, model half precision and shape-changing
  private `_apply` callbacks remain unsupported.
- `no_grad`, `enable_grad`, `set_grad_enabled` and `is_grad_enabled` cover nested
  contexts, direct setters, bare/parenthesized decorators and recursive calls.
  Generator decorators handle send/throw/close/return and restore the caller's
  mode between yields and on errors. Explicit factory `requires_grad` arguments
  keep their native exception to `no_grad`; forward AD is not disabled by it.
  Mode state belongs to the isolated numerical runtime, not an individual thread.
- Model conversion preserves Parameter storage-version counters and links other
  live counters weakly. Later mutations invalidate saved graphs across rebinding.
  Discarded models/arrays are not retained by those links. Temporary conversion
  counters disappear; ordinary converted-model updates return to the unlinked
  counter path. Counter objects use slots rather than instance dictionaries.
- Graph edges retain their original gradient accumulation dtype. Existing
  gradient buffers are updated in place, preserving their object, storage and
  dtype. Native backward after model conversion depends on saved-self, saved-view
  and saved-output rules: some operators use converted inputs, some retain saved
  arrays, and some reject mixed dtypes. Tests cover these distinctions, including
  underflow, reductions, linear algebra and matmul ranks/gradient flags. They do
  not establish unrestricted compatibility for arbitrary old live graphs.
  Convert a model before building its training graph.
- Model conversion does not convert optimizer state. SGD's native mixed-float
  state/gradient cases are supported; Adam retains matching-dtype validation.
  In-memory optimizer loading casts moments to the destination parameter dtype.
  Hooks, native file checkpoint serialization, higher-order reverse AD and a
  production GPU backend remain separate roadmap work.

Square now uses a single saved-self derivative edge rather than multiplying a
Tensor by itself through two generic binary edges. Its derivative avoids
materializing `data ** 1`; original edge dtype and version checks remain intact.
The zero-exponent derivative avoids `0 * inf` at zero inputs.

## Validation

- Synced with main through `505485e`, whose intervening changes concern the UI.
  All **25 candidate Python source hashes** match their benchmark freeze after
  rebasing. No verification producer was live during the rebase.
- Full post-rebase Python suite: **1,704 passed**, one optional Docker test
  skipped, five warnings originating in native Torch (deprecated JIT and native
  generator throw calls), 142.69 seconds. Final focused core suite: **325 passed**.
- New native-reference files contain **115 model-conversion cases and 12
  gradient-mode cases**. They include shared objects/buffers, child conversions,
  gradient identity, CPU overloads/layouts, tiny-value saved-self rules, version
  guards, mixed optimizer state and rank-dependent matmul backward behavior.
- Native lesson reference/mutation/determinism gate: **69 modules, three repeats,
  8,829 test executions and all 295 mutants caught**, 110.54 seconds.
- Real **Pyodide 0.29.4 / NumPy 2.2.5**: five full reference/mutant/learner suites
  (schedules, attention, LoRA, AdamW, Muon split), existing numerical/multifile/
  typed-fallback regressions, and **28 independent contracts**: six module,
  12 optimizer and ten conversion/gradient-mode tests. The maintained WASM
  checker now includes all three standalone contract files on exported sources.
- **Chrome 155.0.8059.40**, five fresh Workers: **35 original tests, all 20
  mutants caught, 28 additional contracts**; package version 0.5.0 and actual
  `/judge/paperleet_torch` source imports confirmed.
- Installed independent wheel outside the checkout in isolated Python with
  **NumPy 1.26.4 and no native Torch: 46 tests passed**. Wheel size **71,844 bytes**;
  SHA-256 `73bdba11e2055056c8a8af745c576fbbc2904a817f02ea5c8dc4441f3da685d1`.
  Full Python packaging tests also cover the disjoint application/private and
  standalone layouts and their export/launcher paths.
- Full current-main web checks, production build and maintained real WASM checker
  passed. Only this worktree's untracked, unreachable intermediate assets are
  eligible for cleanup; tracked published assets are retained.

The package license is still undecided. This is a repository integration
milestone; no PyPI publication is performed.

## Measurements

Baseline: frozen `d7ddd29` / 0.4.0; candidate: frozen 0.5.0 sources. Versions run
sequentially in fresh processes/Workers, with no correctness jobs running during
timing. Identical seeded fixtures use three warmups then **11 × 120 calls** per
case and dtype. CPU is Python 3.12.11 / NumPy 2.5.4 with OMP/MKL/OpenBLAS threads
set to one. Node v25.9.0 and Chrome use the pinned WASM runtime above. Imports and
fixture construction are excluded except for explicitly named construction cases.

| float32 workload | CPU 0.4 → 0.5 (ms) | Chrome WASM 0.4 → 0.5 (ms) |
| --- | ---: | ---: |
| Slice + square backward | 0.05537 → 0.04348 | 0.13083 → 0.11750 |
| Repeated advanced-index backward | 0.10860 → 0.10326 | 0.23250 → 0.22333 |
| Linear + square backward | 0.04626 → 0.04320 | 1.39167 → 1.38417 |
| Attention + square backward | 0.22306 → 0.22033 | 2.69667 → 2.68333 |
| Small MLP training step | 0.06832 → 0.06630 | 0.13667 → 0.13417 |
| Same-dtype Tensor.to | 0.000205 → 0.000115 | timer-quantized |
| Already-contiguous Tensor.contiguous | 0.002008 → 0.000076 | timer-quantized |
| Actual transposed contiguous copy | 0.003877 → 0.004120 | 0.006667 → 0.007500 |

Slice/square backward improves about **21% on CPU**, **10–12% in Node WASM** and
**9–10% in Chrome**, across float32/float64. The single derivative edge and avoided
temporary power array contribute to this result. The small MLP's reset/forward/
backward/SGD step improves about 1–3%; attention and optimizer controls remain
approximately flat. This is not an across-library speedup.

Packing copies cost more: about 6–9% CPU and 13–25% Chrome in this small fixture
(roughly 0.2–1.7 microseconds). Richer layout/copy validation remains measurable;
no packing speedup is claimed. Already-contiguous calls now avoid allocation.
Sub-microsecond browser cases sometimes measure zero due to timer quantization;
their raw samples are retained, and no precise browser ratio is inferred.

An initial candidate regressed CPU slice backward. Profiling identified the
unnecessary first-power derivative allocation; initial and final source/timing
records are retained separately in the local evidence directory. The final
measurements above use the optimized source freeze, whose hashes match the
post-rebase source.

### Allocations

CPython tracemalloc includes registered NumPy allocations, not RSS or browser
heap. Attention float32 forward retained bytes are **201,910 → 201,662**, backward
peak **636,134 → 635,022**; float64 stays similarly flat. Original-edge dtype
metadata does not introduce retained logits or a significant attention peak.
Adam's lazy-state fixture retains zero constructor moment bytes and **524,292 /
1,048,580 bytes** after one parameter's gradient (float32/float64), unchanged from
0.4. A converted four-Parameter MLP discards temporary conversion counters; all
four counters return to the unlinked fast path on its first optimizer update.

Full sample arrays, source hashes, environments, memory, wheel and validation
metadata are in [iteration-5.json](../../packages/paperleet-torch/benchmarks/results/iteration-5.json).
Developer commands are in [benchmarks/README.md](../../packages/paperleet-torch/benchmarks/README.md).
Complete local logs, frozen sources and fresh-Worker scripts are under
`paperleet-wasm-evaluation-20261015/torch-iteration-5`.

Schedulers/hooks, further core tensor/nn coverage, allocation/cold-start profiling
and resident WebGPU storage/graph execution remain in the
[roadmap](../../packages/paperleet-torch/ROADMAP.md). The full independent-library
objective remains active beyond this milestone.

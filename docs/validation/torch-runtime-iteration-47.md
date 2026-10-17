# Torch runtime iteration 47 — InstanceNorm

Version 0.47.0 adds independent per-instance normalization to the standalone and
private PaperLeet runtimes. Main adoption is recorded separately after local gates.
The broader independent-library, efficiency and optional GPU-backend goal remains active.

## Behavior and implementation

- Add `nn.InstanceNorm1d/2d/3d`, `nn.functional.instance_norm`, the low-level
  `torch.instance_norm` entry and ordinary `torch.nn.modules.instancenorm` aliases.
  Batched/unbatched inputs, optional affine parameters and `bias=False` work.
- Extract shared module state into `_NormBase`; InstanceNorm inherits that base
  rather than BatchNorm. Existing BatchNorm adds its native keyword-only `bias`
  option while retaining its counter and cumulative-average behavior.
- Normalize each sample/channel over spatial axes. Update running means and
  unbiased variances across samples, with native buffer-version increments.
  InstanceNorm does not increment `num_batches_tracked`; `momentum=None` means
  zero update momentum, rather than BatchNorm's cumulative average.
- Direct NumPy math and one shared analytical VJP avoid repeated affine tensors
  and extra graph nodes. JVP and reverse-over-forward use native saved-statistic
  conventions. Evaluation statistics and affine values are snapshotted.
- Contiguous input storage is guarded; noncontiguous inputs are copied before
  normalization. Later original affine/running-buffer edits preserve backward
  values. Detached output edits do not invalidate this operator's own tape.
- Retain native functional spatial-singleton errors, low-level singleton behavior,
  empty/partial-buffer state updates, zero-channel affine errors, epsilon edge
  behavior, shared running storage, version-2 and legacy checkpoint handling.

## Local evidence

| Check | Result |
|---|---|
| Complete repository and library CPU suite | 38,714 passed, 1 skipped; 325.05s |
| Focused independent InstanceNorm tests | 182 passed |
| Native PyTorch 2.14.1 differential | 54 numeric + 64 state/boundary cases; 2,142 comparison assertions |
| Node/Pyodide 0.29.4, NumPy 2.2.5 | 5,085 contracts across 44 suites |
| Actual Chrome 155.0.8059.82 Worker | 5,085 contracts, same exact source/test hashes |
| Installed standalone wheel, Python 3.10.19 / NumPy 1.26.4 | 5,098 passed; native Torch absent |
| Standalone/private wheels and source archives | Exact 52 source/license resources, disjoint installation paths |
| Both installed packages, independent uninstall | Alias, checkpoint/optimizer resume, InstanceNorm JVP/VJP pass |
| Original native PaperLeet catalog | 78 problems, 10,218 executions, 342 mutants, three repeats; 220.026s / 600s |
| Complete exported browser WASM catalog | 78/78, 342 mutants, 5,085 contracts; 276.063s |
| Maintained frontend and both build profiles | Five successful local receipts |
| Release policy | Seven files byte-identical to canonical `13d2782a494415ba72bc604a0ef7e4c4f875ba25` |

The native oracle and candidate run in separate Python processes. Numeric cases
check float32/64 C/F/strided outputs, all affine/input VJPs, JVPs and their reverse
pullbacks, plus state values/versions. A separate native probe checks mutation
snapshots, input guards, empty/partial state and epsilon boundaries. Independent
NumPy and finite-difference tests cover every active-gradient subset, retained
backward, module lifecycle and a convolution/InstanceNorm/GELU/SGD checkpoint resume.

The maintained exported-catalog driver runs at concurrency two. The original
native catalog is capped at two numerical-library threads; its runtime is not a
speedup comparison against earlier uncapped runs. No browser budget is exceeded.

## Performance

All figures compare direct InstanceNorm with the existing BatchNorm composition
in the same final source; 0.46 did not expose an InstanceNorm API. Four fresh
processes for each CPU/Node/Chrome runtime, alternating method/runtime order,
five samples per flow and 12 ordinary calls (three SGD steps) retain every sample.
Each child checks values, JVPs, reverse-over-forward and three matching SGD updates
before timing. The 26 flows cover 1D/2D/3D, two dtypes and full convolution training.

Positive percentages mean less elapsed time. Each row aggregates the per-flow
medians of the four fresh children; ranges retain regressions and variation.

| Runtime | Flow | Median time reduction | Range across shapes/dtypes |
|---|---|---:|---:|
| NumPy CPU | forward | 20.7% | 3.6% to 25.0% |
| NumPy CPU | reverse | 26.7% | 5.7% to 30.5% |
| NumPy CPU | jvp | 17.4% | 1.8% to 20.2% |
| NumPy CPU | cnn_sgd | 4.5% | 2.8% to 6.2% |
| Node/Pyodide | forward | 20.3% | 2.6% to 32.8% |
| Node/Pyodide | reverse | 26.3% | 5.0% to 29.2% |
| Node/Pyodide | jvp | 18.4% | 4.3% to 24.0% |
| Node/Pyodide | cnn_sgd | 4.0% | 0.4% to 7.6% |
| Chrome Worker | forward | 20.7% | 5.2% to 30.8% |
| Chrome Worker | reverse | 26.8% | 1.5% to 30.0% |
| Chrome Worker | jvp | 16.7% | 3.6% to 19.4% |
| Chrome Worker | cnn_sgd | 3.4% | 0.0% to 6.9% |

This is a shared-host, same-runtime composition comparison. It does not establish
native PyTorch or GPU speed, peak-memory savings, cold-start behavior, or uniform
improvements for other shapes. No WebGPU execution or backend is claimed here.

## Identity, packaging and integration

The candidate has 50 Python modules plus LICENSE/NOTICE: 48 old resources remain
unchanged, three old resources change (`__init__.py`, `_api.py`, `_batchnorm.py`),
and `_instancenorm.py` is added. Sorted source/license bytes change from
610,370 to 622,660;
local concatenated gzip changes from 142,106 to
144,376. These are source-footprint
estimates, not wheel/network bytes or measured process memory.

All 10,387 previously published immutable payloads are preserved; 158 new payloads
are reachable, yielding 10,545 total. The 161 active export identities match the
final source. Public/private wheel and source archive payloads match that source;
both packages work outside the checkout without native Torch. No dependency or
lockfile changes are included. Existing owned runtime and lockfile hashes were
rechecked before the exported-browser run.

Float16/complex normalization, general differentiable non-leaf mutation, storage
rebinding across saved normalization graphs and a persistent WebGPU tensor backend
remain outside this version's validated support. Floating-point accumulation is
not a bitwise equivalence promise for ill-conditioned or exceptional input.

Evidence: `packages/stridetorch/benchmarks/results/iteration-47.json`,
`iteration-47-integration.json` and four matching raw harness archives preserve
source/test identities, native reference/candidate outputs, all timing samples,
command receipts and logs. `bench_instance_norm.py` is the portable public benchmark.
Prototype runs are not counted as final verification; an unversioned partial CPU
run was explicitly stopped after source changes and replaced with the frozen run.

No tag, workflow dispatch, release, PyPI upload or deployment is initiated by this work.

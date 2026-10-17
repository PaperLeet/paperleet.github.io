# Torch runtime iteration 49 — Bounded regular-convolution padding

Version0.49.0 fixes avoidable full padded activation/adjoint allocation in regular
zero-padded Conv1d/2d/3d and shares its clipped primitives with transposed convolution.
Main adoption is recorded after local gates. The broader independent-library,
core API/efficiency and optional GPU-backend goal remains active.

## Implementation and support

- Add shared `_convolution_padding` gather, clipped scatter and per-tap kernels.
  Regular and transposed convolution reuse clipped coordinate and gather rules;
  transposed forward keeps its established inline scatter hot path.
- Auto keeps the existing padded path for modest zero-padding spans. A span over
  32MiB or four times the sampled-patch size selects bounded execution. A full
  packed gather over32MiB selects per-tap fallback, including when matrix work
  was requested. Reflect/replicate/circular padding retain their previous paths.
- Matrix selection uses the actual input-gradient extent for clipped execution.
  Forward gathering is packed directly in contraction order. Expanded gathered
  arrays are released after forward; weight gradients regather from the saved
  original input rather than retaining expanded activations on the tape.
- A bounded64-entry LRU stores only immutable geometry decisions, including
  workspace limits. No tensor, seed, activation or gradient values enter it.
- Input VJPs scatter into the actual input shape. Per-tap forward and weight
  gradients retain implicit zero products for nonfinite operands/seeds; finite
  paths skip only mathematically zero regions. Existing version/rebinding guards
  and bias-only input/weight object release remain in force.
- Preserve float32/64, groups, unbatched/strided inputs, asymmetric same padding,
  per-axis stride/dilation, all gradient subsets, retained tapes, bilinear JVP
  and reverse differentiation of JVPs. Existing transposed-convolution output
  layouts, output-size resolution, empty metadata and checkpoint behavior remain.

These are logical per-buffer estimates and tested allocation policies, not a
process-RSS/BLAS-internal peak-memory guarantee. The bounds apply to zero-padding;
nonzero padding modes do not gain them. Public APIs, dependencies and dtype limits
are unchanged. New controls are private validation/benchmark overrides.

## Local evidence

| Check | Result |
|---|---|
| Complete repository/library CPU suite | 39,521 passed, 1 skipped; 433.00s |
| New bounded-padding tests | 292 passed |
| Existing regular/transposed convolution regression | 207 +515 passed; 1,014 focused total |
| Native regular-convolution comparison | 96 cases ×6 routes; 6,912 array/layout checks |
| Native transposed-convolution comparison | 144 cases ×5 routes; 8,640 array/layout checks |
| Native nonfinite regular-padding comparison | 12 cases ×6 routes; 216 value/VJP checks |
| Native transposed state/layout/IEEE/module comparison | 68 cases; 22,436 comparisons |
| Node/Pyodide0.29.4 / NumPy2.2.5 | 5,892 contracts, 46 suites |
| Actual Chrome 155.0.8059.82 Worker | 5,892 contracts, exact source/test identities |
| Installed standalone wheel, Python3.10.19 / NumPy1.26.4 | 5,905 passed; native Torch absent |
| Public/private wheels and source archives | Exact54 source/license resources, disjoint installed paths |
| Both packages, independent uninstall | Aliases, checkpoints, ordinary/transpose JVP/VJP and huge-halo Conv3d |
| Original native PaperLeet catalog | 78 problems, 10,218 executions, 342 mutants, three repeats; 277.475s /600s |
| Complete exported WASM catalog | 78/78, 342 mutants, 5,892 contracts; 280.144s |
| Latest frontend and both build profiles | Five successful local receipts |
| Release policies | Seven files byte-identical to canonical `13d2782a494415ba72bc604a0ef7e4c4f875ba25` |

Native oracle is PyTorch2.14.1 in separate Python processes. Ordinary routes are
auto, gather+direct/matrix, per-tap, zero-workspace fallback and a safe legacy padded
control. Numeric cases check values, VJPs, JVPs, reverse-over-forward and strides.
Float32 tolerances are rtol2e-5/atol3e-5; float64 uses8e-12 for regular and5e-12 for
transposed cases. Nonfinite values use equal-NaN checks. Existing mutation failures
retain the adapter's RuntimeError-compatible typed unsupported subtype.

Independent scalar loops and finite differences check bounded math. Allocation
regressions forbid the full padding and padded-adjoint helpers for 1D/2D/3D
padding/dilation1,000,000. A zero-gather-workspace case forbids both gather and
padding helpers. A weak-reference test proves forward gathered patches are freed
and regenerated for repeated backward. Huge legacy spans are not executed merely
to demonstrate their allocation failure. The original native catalog requests two
numerical-library threads; exported WASM uses concurrency two and remains within
every browser budget.

## Performance

The formal comparison uses actual0.48 and0.49 source snapshots plus same-source
padded, gather and per-tap controls. Four fresh processes per CPU/Node/Chrome use
alternating method/runtime order, five samples and eight ordinary calls or three
encoder/decoder SGD steps. Every child validates values, VJP/JVP/mixed pullbacks
and three matching optimizer updates before timing; actual strategy/matrix
selection is qualified. All32 flows and every sample are retained.

Legacy comparisons use safe 1D/2D/3D halos (65,536/127/31), small real input/output
extents, and ordinary/transpose controls. Training composes large-halo Conv3d,
InstanceNorm/GELU, ConvTranspose3d, loss and SGD including the latent input gradient.
Positive percentages mean reduced time versus0.48; ranges retain all measured
variation and regressions.

| Runtime | Flow | Median time reduction | Range across shapes/dtypes |
|---|---|---:|---:|
| NumPy CPU | forward | 82.4% | 74.1% to 91.5% |
| NumPy CPU | reverse | 76.0% | 61.4% to 87.8% |
| NumPy CPU | jvp | 81.8% | 71.4% to 94.2% |
| NumPy CPU | encoder_decoder_sgd | 55.3% | 53.9% to 56.7% |
| Node/Pyodide | forward | 82.8% | 73.6% to 86.1% |
| Node/Pyodide | reverse | 76.3% | 63.2% to 82.1% |
| Node/Pyodide | jvp | 83.1% | 72.3% to 88.7% |
| Node/Pyodide | encoder_decoder_sgd | 43.0% | 39.4% to 46.6% |
| Chrome Worker | forward | 84.1% | 71.6% to 85.6% |
| Chrome Worker | reverse | 75.5% | 65.6% to 82.3% |
| Chrome Worker | jvp | 83.0% | 72.5% to 88.9% |
| Chrome Worker | encoder_decoder_sgd | 46.5% | 44.7% to 48.2% |

Controls cover forward/reverse/JVP in both dtypes:

| Runtime | Control | Range of time reduction |
|---|---|---:|
| NumPy CPU | ordinary | -7.9% to 7.4% |
| NumPy CPU | transpose_control | -7.4% to 2.2% |
| Node/Pyodide | ordinary | -4.0% to 2.8% |
| Node/Pyodide | transpose_control | -4.3% to 0.6% |
| Chrome Worker | ordinary | -6.9% to 2.9% |
| Chrome Worker | transpose_control | -7.4% to 3.9% |

One Chrome child failed while fetching the pinned pytest dependency before any
benchmark code/timing ran. Its failure receipt/log are preserved; only that child
was retried, and the remaining scheduled children completed. Eight already
completed children were reused with their exact source/sample checks. There are
12 successful timing groups plus that separate dependency-only failed attempt.
The transpose `_primal` AST is identical to0.48 after restoring the inline hot
path; helper definitions differ and have their own native/regression coverage.

This is shared-host, same-runtime evidence. It does not establish universal speed,
native Torch/GPU speed, cold-start improvement or process peak-memory reduction.
No GPU execution is claimed. Ordinary workloads may incur selector overhead or
noise; controls and same-source padded measurements are retained explicitly.

## Identity and integration

The candidate has52 Python modules plus LICENSE/NOTICE. Fifty old resources are
unchanged; `__init__.py`, `_convolution.py` and `_transpose_convolution.py` change,
and `_convolution_padding.py` is added. Sorted source/license bytes change from
641,077 to 647,717;
local concatenated gzip changes from 148,684
to 150,068. These are source estimates,
not wheel/network bytes or measured memory.

All10,703 prior immutable payloads are preserved. The158 new reachable payloads
yield10,861 total;161 active export identities match the final source. Public and
private packages work outside the checkout without native Torch. Fourteen pinned
Pyodide files and the npm lock identity were rechecked, with no dependency/lock
changes. Concurrent upstream frontend work is preserved as exact Git blobs; its
fresh frontend/build receipts and source changes are recorded in `upstream_syncs`.
The frozen numerical sources/tests/exports and private Python resources are
unchanged across those frontend-only updates.

Float16/complex convolution, general non-leaf mutation, saved-storage rebinding,
nonzero-mode padding bounds and a persistent GPU backend remain outside this
version's new support. Numerical accumulation is not a bitwise equivalence promise.
Broader core API coverage and efficiency work continues toward independent-library quality.

Evidence: `packages/stridetorch/benchmarks/results/iteration-49.json`,
`iteration-49-integration.json` and four matching raw harness archives retain
source/test hashes, separate oracle outputs, allocation/lifetime tests, command
receipts, logs and timing samples. `bench_convolution_padding.py` is the portable
public benchmark. Initial control regressions, pre-cache source and all initial timing samples are
retained in `control-history` in the raw harness. The first formal attempt was
replaced after hot-path changes; final claims use the new complete frozen run.

No tag, workflow dispatch, release, PyPI upload or deployment is initiated by this work.

## Main adoption

Validated source `7b5e8f78886a64d00caf831ae53d53cf26346be2` was adopted on remote main by ordinary non-force push. The remote SHA was verified exactly; zero Actions runs were observed for this source SHA. All 54 source/license resources, 161 active export identities, 10,861 immutable payloads and seven canonical release-policy files were rechecked. The separate proof commit is verified outside the repository after its push. No tag,dispatch,release,PyPI upload or online deployment was initiated by this work.

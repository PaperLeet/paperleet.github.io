# Torch runtime iteration 48 — Transposed convolution

Version 0.48.0 adds grouped transposed convolution to the standalone and private
PaperLeet runtimes. Main adoption is recorded after local gates. The broader
independent-library, efficiency and optional GPU-backend goal remains active.

## Behavior and kernels

- Add `nn.ConvTranspose1d/2d/3d`, their functional and root `torch` entries, plus
  ordinary `torch.nn.modules.conv` imports and the `_ConvTransposeNd` base alias.
  Batched/unbatched inputs, groups, per-axis stride/padding/dilation, output
  padding, optional bias, module `output_size`, conversion and checkpoints work.
- Generalize `_ConvNd`'s existing state/reset infrastructure for the transposed
  weight layout `(in_channels, out_channels // groups, *kernel)` and native
  output-channel fan-in. Existing regular convolutions retain their reset values.
- Direct clipped scatter and packed matrix contraction share convolution helpers
  for analytical input/weight/bias VJPs. Bilinear JVPs retain reverse derivatives
  and explicit zero terms, including native nonfinite behavior.
- Preserve input/weight version guards even for bias-only tapes; detached bias
  and output writes do not invalidate the operator's saved values. Unneeded
  input/weight objects are released on bias-only tapes.
- Output padding is smaller than either stride or dilation, per axis; it can
  exceed stride when dilation is larger. Backward windows are limited to the
  actual input points. Module `output_size` uses the native stride-sized range
  and accepts spatial dimensions or a full output shape.
- Empty-batch output and tangent strides are conventional. This also corrects
  zero-stride empty output/tangent metadata in existing regular convolution.
  Batched 2D channels-last input and 2D channels-last weights select the checked
  output layouts, including singleton channels and unbatched weight-driven layout.

## Allocation policy

The forward path scatters only into the actual output, without a full theoretical
padded activation. Backward uses a strided padded view for modest spans; it instead
gathers directly into packed order when that span is excessive. The direct
fallback evaluates per-tap slices. Native implicit padded-zero products remain
present for nonfinite backward values, including `0 * Inf -> NaN`.

Optional matrix work uses a conservative 32 MiB logical-array estimate. A separate
32 MiB padding threshold and a padded-versus-gathered size comparison prevent
large unnecessary zero buffers. The large-dilation test forbids the padding helper:
3D dilation/padding 10,000, and nonfinite 1D dilation/padding 10,000,000, succeed
without allocating those full theoretical spans. This is a tested allocation
policy, not a process-RSS or BLAS-internal peak-memory measurement.

CPU and actual WASM pilots retained forced direct/matrix controls. They showed
that grouped image and volume workloads benefit from selecting matrix work at a
smaller threshold than ordinary convolution. Auto keeps point kernels and tiny
work direct, uses matrix work from 8,192 logical multiply terms, and retains the
workspace/finite-value guards. All formal samples and selection qualification
counts are retained; no universal fast-path claim is made.

## Local evidence

| Check | Result |
|---|---|
| Complete repository/library CPU suite | 39,229 passed, 1 skipped; 442.04s |
| Focused transposed/regular convolution | 515 + 207 = 722 passed |
| Native PyTorch 2.14.1 differential | 144 numeric cases × 5 routes, 8,640 array/layout checks |
| Native state/layout/IEEE/module comparison | 68 cases, 22,436 numeric comparisons; RuntimeError base-class behavior |
| Node/Pyodide 0.29.4 / NumPy 2.2.5 | 5,600 contracts, 45 suites |
| Actual Chrome 155.0.8059.82 Worker | 5,600 contracts, exact source/test identities |
| Installed standalone wheel, Python 3.10.19 / NumPy 1.26.4 | 5,613 passed, native Torch absent |
| Standalone/private wheels and source archives | Exact 53 source/license resources, disjoint installed paths |
| Both installed packages and independent uninstall | Aliases, InstanceNorm and transposed-convolution JVP/VJP, checkpoint/optimizer resume |
| Original native PaperLeet catalog | 78 problems, 10,218 executions, 342 mutants, three repeats; 177.123s / 600s |
| Complete exported WASM catalog | 78/78, 342 mutants, 5,600 contracts; 398.571s |
| Latest frontend and both build profiles | Five successful local receipts |
| Canonical release policies | Seven files byte-identical to `13d2782a494415ba72bc604a0ef7e4c4f875ba25` |

Independent scalar scatter loops and finite differences cover values, all active
VJP subsets, JVPs and their reverse pullbacks. An adjoint identity checks regular
convolution against transposed convolution. Cases include groups, anisotropic
geometry, output-padding-beyond-stride, cropped taps, empty batches, native dtype
boundaries, saved versions, layout, parameter identity/reset and output-size limits.
A complete transposed-convolution/InstanceNorm/GELU/Conv2d decoder resumes AdamW
training from a serialized model/optimizer checkpoint.

Native and candidate run in separate processes. Numeric routes are auto, forced
direct/matrix and both forced paths with padded workspace disabled. Float32 uses
rtol 2e-5 / atol 3e-5; float64 uses 5e-12. State/boundary comparisons use float64
and equal-NaN handling. The adapter keeps its typed unsupported subtype for saved
version failures; it remains catchable as RuntimeError, as checked by the oracle.

The original native catalog requests two numerical-library threads. The exported
WASM catalog uses concurrency two and stays within every browser budget. These
whole-catalog runtimes are verification measurements, not speedup claims against
previous releases or other runtimes.

A separate source/geometry audit identifies an existing follow-up in regular
convolution: its large-padding path still materializes the full padded span.
The allocation bounds above apply to transposed convolution. The next priority
is sharing bounded gather/scatter with ordinary convolution; potentially huge
ordinary padded arrays were not executed during this audit. The exact geometry
and source findings are retained in `next-iteration-candidates.json` in the raw
harness evidence.

## Performance

The baseline is same-source zero insertion followed by grouped, spatially flipped
regular convolution. Version 0.47 did not expose transposed convolution. Inputs
are finite; that composition is not an IEEE oracle for implicit padding products.
Forced direct and matrix controls are retained alongside auto.

Four fresh processes per CPU/Node/Chrome runtime use alternating method/runtime
order, five samples and 12 ordinary calls (three decoder SGD steps). Each child
checks values, JVPs, reverse-over-forward and three matching SGD updates before
timing. The 38 flows cover sequence/image/grouped/volume/larger-image/point
convolutions in two dtypes and complete decoder training. Actual auto/direct/matrix
selection is qualified before primitive timing. All raw samples remain available.

Positive percentages mean reduced elapsed time versus the composition. Rows
aggregate per-flow medians of four independent children; ranges retain noise and
any regressions.

| Runtime | Flow | Median time reduction | Range across shapes/dtypes |
|---|---|---:|---:|
| NumPy CPU | forward | 51.4% | 23.0% to 66.7% |
| NumPy CPU | reverse | 39.8% | 29.8% to 63.2% |
| NumPy CPU | jvp | 46.8% | 18.3% to 65.0% |
| NumPy CPU | decoder_sgd | 19.9% | 18.2% to 21.6% |
| Node/Pyodide | forward | 51.1% | 33.7% to 74.7% |
| Node/Pyodide | reverse | 45.3% | 27.9% to 70.7% |
| Node/Pyodide | jvp | 49.8% | 31.0% to 74.1% |
| Node/Pyodide | decoder_sgd | 10.0% | 8.9% to 11.1% |
| Chrome Worker | forward | 52.1% | 36.9% to 74.9% |
| Chrome Worker | reverse | 45.4% | 27.3% to 70.8% |
| Chrome Worker | jvp | 50.0% | 31.2% to 74.3% |
| Chrome Worker | decoder_sgd | 7.5% | 4.3% to 10.6% |

These are shared-host, same-runtime measurements, not native PyTorch/GPU speed,
peak-memory or cold-start measurements. No WebGPU execution is claimed.

## Source, packaging and current main

The candidate has 51 Python modules plus LICENSE/NOTICE. Forty-nine old resources
are unchanged; `__init__.py`, `_api.py` and `_convolution.py` change, and
`_transpose_convolution.py` is added. Sorted source/license bytes change from
622,660 to 641,077;
local concatenated gzip changes from 144,376
to 148,684. These are source estimates,
not wheel/network transfer sizes or process memory.

All 10,545 previously published immutable payloads are preserved. The 158 new
reachable payloads yield 10,703 total; 161 active export identities match the final
source. Public/private package paths are disjoint and work outside the checkout
without native Torch. Fourteen pinned Pyodide files and the npm lock identity
were rechecked; no dependency/lock changes are included.

Concurrent upstream frontend updates are preserved as exact Git blobs. Their
source changes and five local frontend/build receipts are recorded in
`upstream_syncs`; the frozen numerical source, tests, exports and private package
resources remain unchanged. Updated frontend files are not included in the private
Python source archive. Main adoption must preserve the latest verified upstream
head and avoid an active quality/Pages chain.

Float16/complex convolution, general non-leaf mutation, rebinding saved tensor
storage and a persistent GPU backend remain outside validated support. Bitwise
agreement for arbitrary ill-conditioned/nonfinite data is not promised. Broader
core API and backend work remains active toward independent-library quality.

Evidence: `packages/stridetorch/benchmarks/results/iteration-48.json`,
`iteration-48-integration.json` and four matching raw harness archives preserve
source/test identities, commands/logs, independent oracle outputs, all samples and
qualified algorithm choices. `bench_transpose_convolution.py` is the portable
public benchmark. Prototypes and initial test-oracle failures are excluded from
formal claims and retained as exploratory evidence.

No tag, workflow dispatch, release, PyPI upload or deployment is initiated by this work.

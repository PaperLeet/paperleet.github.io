# StrideTorch 0.38: max pooling with measured NumPy kernels

Version 0.38 adds `nn.MaxPool2d`, functional `max_pool2d` and
`max_pool2d_with_indices`, plus the values-only root `torch.max_pool2d` entry
point. Existing `import torch` aliases and launcher behavior remain available.
The independent NumPy package retains its current name while the user's
browser-oriented naming decision is pending.

## Values, indices and derivatives

The implementation handles CHW/NCHW input, noncontiguous arrays and empty
batches. Kernel/stride/padding/dilation tuples, default and empty stride, ceil
rounding and dropped windows starting entirely in right padding follow the
native-reference probes. Functional and Module argument orders retain their
respective public conventions. The pooling child namespace also supports
ordinary `from torch.nn.modules.pooling import MaxPool2d`.

Returned indices are int64 spatial offsets within each input channel. Ordinary
finite ties select the first input sample; NaNs select the last NaN encountered.
Padding cannot win an all-negative-infinity or integer-minimum tie over a real
input. Selected zero signs are preserved. Overlapping pullbacks accumulate in
native output order; nonoverlapping updates still add into zero so negative-zero
seeds have the native gradient sign. Tests cover float16/float32/float64 and
int8/int16/int32/int64/uint8 inference. Bool/complex and other unavailable modes
retain typed unsupported failures. Float16 autograd remains outside the
library's existing real float32/float64 AD boundary.

Float32/float64 support VJP, retained backward, JVP and reverse-over-forward.
The primal pullback saves input/index storage versions and does not save output
values. The tangent pullback saves tangent/index versions, with no false reverse
edge to the primal input. Detached output writes are allowed; saved input/index
writes invalidate the relevant tape. Existing restrictions on direct mutation
of differentiable non-leaf tensors still apply. Geometry is snapshotted and
no sampling Jacobian is materialized. Module state/training/forward hooks,
standard imports, pickle and graph ownership are covered independently.

Extreme dilated/padded windows may have no real sampled input. Actual native
probes return negative infinity and an index outside the input for examples
such as H=W=1, kernel=2, padding=1, dilation=2. Inference reproduces these values
and indices. Derivatives for that geometry are explicitly unsupported rather
than gathering/scattering invalid indices. This is a documented boundary, not a
claim that every native pooling mode or general higher-order reverse AD works.
Public geometry/parameter descriptions are available in the official
[MaxPool2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MaxPool2d.html)
and [functional documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.max_pool2d.html);
selection, mutation and AD details also have recorded native CPU probes.

## Kernel choice and bounded packing

The first prototype compared generic offset scanning and vectorized packed
windows. Pure window selection helped large kernels and index/AD paths, but
slowed common 2x2 CNN inference on both CPU and WASM. Those original sources,
measurements and both refined pilots are retained.

The default uses a direct 2x2 path for unpadded, unit-dilation, stride-two windows
without a partial ceil tail. Other index/AD paths and larger inference kernels
can use window packing. A conservative 32 MiB temporary-workspace estimate
limits packing; other cases scan the kernel positions. Both implementations
share first-tie/last-NaN semantics. Unpadded windows avoid unnecessary coordinate
validation and inference avoids retaining full spatial indices. All private
algorithm overrides, including fallback cases, are covered against the same
oracles; they are not public API parameters.

This change retains NumPy CPU/Pyodide WASM execution and introduces no dependency
or GPU backend. The measurements below establish benefits for the tested flows;
they do not establish a WebGPU benefit or cross-runtime/native-Torch throughput.
A GPU route would need a complete resident tensor/AD path and a measured
end-to-end advantage, including transfers, before becoming a default backend.

## Local correctness and application gates

- Focused native pooling, independent contracts and feature discovery:
  **703 passes**.
- Full merged Python suite: **33,167 passes, 1 optional skip,
  34 warnings, 264.31s**.
- Final random oracle: **450 geometry/dtype trials**, all four algorithm
  selections, with values/indices/signs and valid reverse gradients compared
  against native Torch. Original prototype trials remain separate.
- Actual Node Pyodide and Chrome Worker: **1,730 checks in 37 suites each**,
  pinned Pyodide 0.29.4 / NumPy 2.2.5 without native Torch.
- Independent Python 3.10.19 / NumPy 1.26.4 wheel: **1,743 passes**.
  All **48** source/license resources match public/private wheels and source
  archives. Both installed namespaces run pooling VJP/JVP, module pickle,
  aliases and earlier checkpoint/optimizer workflows, then survive independent
  uninstall without native Torch installed.
- Original native catalog: **78 problems, 10,218 executions, 342 mutants,
  3 repeats, 190.12s** with unchanged deadlines.
- Actual exported WASM: **78/78 modules, 342 mutants,
  249644ms**, concurrency 2, no budget exceeded.
  Its attention test VM also executes all **1,730 checks/37 suites**, supplementing
  only the actual exported Schedules legacy module after production checks.
- Fresh catalog/private packaging, complete frontend checks and both typed
  builds pass on upstream `d06cf1d9e65e8694cf89dddc364f0b7780231556`. All **8,965** prior immutable
  payloads are retained byte-for-byte; **158** new assets are reachable and all
  **161** active index/payload files are SHA-256 frozen. The smaller Schedules
  profile remains six resources with its prior no-AD boundary.

Source comparison finds **45 unchanged shared resources**, one new pooling
module and version/API registration changes. Existing API AST is identical after
removing only the pooling additions. This supports, but does not replace, the
old-function regression measurements and full runtime tests.

The first independent test run incorrectly attempted direct mutation of
non-leaf tensors outside the library's supported mutation API and expected
leaves to disappear while their graph still owned them. Tests now use detached
storage aliases and require collection after discarding the graph, matching the
native ownership/version contract. The first 12 failures remain available.

## Complete new pooling and CNN workflows

Eight sequential fresh children cover two dtypes and two processes per CPU/WASM.
Each workflow uses nine ABBA/BAAB blocks, warmups, GC enabled and equal loop
counts. Every block checks outputs, gradient routes, dtypes and zero signs;
CNN training performs real SGD/momentum updates and checks updated parameters.
The table reports time reduction of the **default automatic path versus the
same version's generic scan**, not a comparison with version 0.37, which lacks
pooling. Negative values mean slower. Raw times, samples, first/refined pilots
and source identities remain in the reports.

| Complete workflow | CPU time reduction | WASM time reduction |
| --- | ---: | ---: |
| tiny-inference | 63.1 to 64.7% | 60.5 to 62.9% |
| tiny-indices | 55.0 to 55.6% | 49.0 to 53.1% |
| tiny-backward | 26.8 to 29.1% | 17.8 to 27.3% |
| tiny-jvp | 23.6 to 25.2% | 20.8 to 23.3% |
| image-inference | 37.5 to 38.3% | 39.0 to 40.2% |
| image-indices | 34.1 to 35.8% | 29.2 to 31.6% |
| image-backward | 19.0 to 21.1% | 17.5 to 21.1% |
| overlap-inference | -2.6 to 2.9% | -2.1 to 3.1% |
| overlap-indices | 18.5 to 24.3% | 22.5 to 33.9% |
| overlap-backward | 13.6 to 21.6% | 24.3 to 27.4% |
| overlap-jvp | 12.5 to 18.4% | 20.3 to 24.5% |
| dilated-inference | -1.1 to 1.5% | -1.8 to 2.5% |
| dilated-indices | 33.1 to 34.1% | 27.4 to 29.8% |
| dilated-backward | 19.8 to 21.0% | 14.1 to 21.9% |
| large-kernel-inference | 32.5 to 34.0% | 29.5 to 33.7% |
| large-kernel-indices | 61.4 to 61.9% | 59.0 to 61.0% |
| large-kernel-backward | 51.7 to 53.2% | 48.5 to 51.4% |
| workspace-fallback-inference | -0.2 to -0.0% | -0.9 to 0.4% |
| workspace-fallback-indices | -1.0 to 0.3% | -0.7 to 0.2% |
| workspace-fallback-backward | -0.4 to 0.1% | -1.4 to 0.9% |
| cnn-tiny-inference | 22.1 to 22.9% | 16.9 to 24.7% |
| cnn-tiny-sgd | 9.3 to 11.6% | 9.7 to 12.7% |
| cnn-image-inference | 16.0 to 17.4% | 8.9 to 17.6% |
| cnn-image-sgd | 7.7 to 10.2% | 7.3 to 9.6% |

## Existing 0.37 to 0.38 workflow controls

A separate eight paired children retain the 21 old inverse/solve/JVP/autograd,
SGD and Schedules flows. Both versions use the same public APIs and the same
Python optional-feature default lookup. This table measures complete-flow
regressions; it does not claim a uniform numerical speedup from adding pooling.

| Existing workflow | CPU time reduction vs 0.37 | WASM time reduction vs 0.37 |
| --- | ---: | ---: |
| tiny-inference | 0.5 to 1.1% | -2.4 to 1.5% |
| small-inference | -0.2 to 0.2% | -1.7 to 0.2% |
| large-inference | -0.8 to 0.6% | -0.6 to 0.2% |
| upper-inference | -0.9 to 1.2% | -0.6 to 0.0% |
| batch-inference | 0.1 to 0.4% | 0.1 to 1.6% |
| tiny-backward | -1.4 to 2.2% | -2.4 to 0.3% |
| small-backward | -0.1 to 2.5% | -0.2 to 2.2% |
| large-backward | -0.3 to 1.8% | -0.4 to 0.5% |
| sgd-factor | -0.5 to 0.5% | -0.3 to 0.7% |
| optional-sgd | 0.0 to 0.8% | -0.3 to 3.4% |
| spd-chain | -1.7 to 0.5% | -0.3 to 1.0% |
| jvp | 0.4 to 1.5% | 0.4 to 1.4% |
| large-jvp | -1.0 to 1.1% | -0.3 to 1.1% |
| jvp-backward | -0.6 to 0.9% | -0.0 to 1.4% |
| large-jvp-backward | 0.3 to 0.9% | -0.3 to 0.8% |
| fixed-factor-jvp | -0.6 to 0.4% | 0.2 to 0.3% |
| legacy-inference | 0.1 to 0.7% | -0.3 to 0.5% |
| legacy-optional-inference | 0.0 to 1.1% | -0.5 to 1.3% |
| generic-control | -1.0 to 1.7% | -1.1 to 1.3% |
| cholesky-control | -2.3 to 0.0% | -0.4 to 1.9% |
| unary-control | -1.9 to 1.8% | -0.4 to 1.1% |

Cold CPU namespace import uses 16 fresh alternating children, reporting four
samples per version with/without preloaded NumPy. Median including NumPy:
96.98 → 97.17ms;
preloaded: 72.36 → 74.23ms.
Source bytes: 525960 → 538761;
local gzip estimates: 123901 →
127080. Shared-host variation
and requested, unverified vendor thread counts limit interpretation. These are
not browser wire sizes, statistical guarantees or native-Torch import/throughput
comparisons. Raw outliers remain in the tables and archives.

## Reproduction and adoption

[`iteration-38.json`](../../packages/torchwasm/benchmarks/results/iteration-38.json)
and [`iteration-38-integration.json`](../../packages/torchwasm/benchmarks/results/iteration-38-integration.json)
contain exact source/test/artifact hashes, child receipts, full measurements and
hashed reproduction archives. Native-reference tests are test oracles, not
runtime dependencies. Later main synchronization/adoption is recorded separately.

The [explicit-release policy](../releasing.md) is unchanged: no CI, tag, workflow
dispatch, package upload or online publication is requested. The user will tag
later. Local results do not establish remote Linux CI or an online deployment.
Other pooling dimensions, average pooling, transpose convolution, normalization
coverage and measured backend improvements remain active work toward independent
library readiness.

Final main synchronization preserves all eight incoming preparation/catalog/UI
files through `f649a1b6362159f1e8055abdc334640f18763d7f` byte-for-byte. A freshly rebuilt private
wheel/source archive contains the current interview catalog, passes source and
public/private ownership checks, and survives coinstall/uninstall. Affected
backend catalog tests pass 25 cases; the complete frontend check and both typed
builds pass again. All 48 computational resources and 161 current exports remain
unchanged. The full numerical/WASM gates above retain their original source and
export identities. See `iteration-38-main-sync.json`.

Validated source was adopted non-force on remote `main` at `082e8880eb23b06566ba8d430f16857602c6adab`.
A read-only query found zero Actions runs for this commit. See
`iteration-38-main-adoption.json`; no tag, dispatch or online release was created.

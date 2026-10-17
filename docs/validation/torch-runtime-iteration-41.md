# Torch runtime iteration 0.41: one-dimensional average pooling

This candidate adds `AvgPool1d` and `AdaptiveAvgPool1d`, functional entry points
and native-style root aliases. CL/NCL inputs reuse the ordered 2D numerical
kernels through NumPy shape views, without constructing extra Tensor view graph
nodes. Public and private distributions remain independent. Main adoption is
recorded separately after local validation.

## Compatibility and computation

Fixed pooling supports integer/one-element kernel, stride and padding arguments,
ceil mode and padding inclusion. Constructor normalization, stored attributes,
repr and validation timing follow the inspected native Torch 2.14.1 implementation.
Divisor overrides remain a 2D option. Float16/float32/float64 inference and fixed
int64 inference are supported; adaptive int64 and half-precision AD remain explicit
unsupported boundaries. Reverse AD, JVP and reverse-over-forward cover float32/64.

Global adaptive length 1 uses a stateless mean: an empty length produces NaNs
and an empty input gradient. Nonglobal adaptive output length 0 is valid forward
but rejects backward. Empty unbatched channels and batched channels have distinct
native backward rules. Saved input and tangent mutation, output edits and the
absence of a false primal edge in the linear JVP are checked against native Torch.

Independent region sums and transpose scatter oracles cover values, VJP/JVP,
finite differences, nonfinite locality and C/Fortran/strided storage through five
internal algorithm combinations. A Conv1d–SiLU–fixed/adaptive pool–flatten–linear
model is checked with momentum SGD, checkpoint restoration and continued training.

Twelve additional native layout probes exposed a pre-existing empty-output stride
mismatch in both 1D and 2D pooling, despite the first candidate passing 35,710
tests. The fix uses the existing conventional-empty-layout helper. Sixty added
standalone/WASM route checks and twelve native checks cover this correction.
The earlier source, test results, twelve mismatching probes and artifacts are
retained in the harness; final results below refer to the refreshed source hashes.

## Validation

- Focused pooling/native/layout tests: 2,615 passed.
- Full repository/library CPU suite: 35,782 passed, one optional skip; 411.63 s.
- Node/Pyodide and actual Chrome Worker: 2,858 independent checks each, 40 suites; Pyodide 0.29.4 / NumPy 2.2.5.
- Isolated Python 3.10.19 / NumPy 1.26.4 wheel: 2,871 passed, native Torch absent.
- Native random oracles: 240 seeded geometries and 1,200 value/gradient/tangent array comparisons, plus constructor, parameter, IEEE, model and empty-shape contracts.
- Public/private wheels and source archives: all 49 source/license resources match; coinstallation, ordinary torch aliases and independent uninstall pass.
- Original native catalog: 78 problems, 10,218 executions, 342 mutants, three repeats, 153.862 s within the unchanged 600 s budget.
- Exported WASM catalog: 78 problems, 342 mutants, 418043 ms, concurrency 2, no exceeded browser budgets. Exported attention includes the 2,858 independent checks against the actual exported source.
- Full frontend, release-policy checks and both typed production profiles pass. The first concurrent frontend syntax subprocess timed out at its maintained five-second deadline; a focused rerun and final sequential full run use the explicitly owned Python. Original failure is retained; no maintained timeout was increased.
- All 9,439 previous immutable payloads are byte-identical; 158 new reachable payloads and 161 active exported identities are frozen.

## Performance

The baseline is the already-supported 0.40 `unsqueeze → 2D pool → squeeze`
composition. It never times an unsupported old 1D API. The candidate executes
both that composition and the direct 1D API. All outputs/gradients and one full
training update are compared before timing. Each CPU, Node/Pyodide and Chrome
Worker runtime has four fresh processes with alternating variant order, seven
samples per flow and 40 calls per sample (five for complete training). The 34
flows include both float32/64, strided inference, VJP, JVP and training. Timings
include dispatch, allocation and graph work. Shared-host variance remains; these
are not native Torch, GPU, transfer or cross-runtime throughput comparisons.

Below are median reductions in elapsed time against the baseline composition;
ranges span the two dtypes after taking the median of four process results.

| Flow | CPU | Node/WASM | Chrome Worker/WASM |
| --- | ---: | ---: | ---: |
| Fixed forward | 19.0–20.4% | 18.1–19.6% | 18.5–26.2% |
| Irregular forward | 16.4–17.7% | 16.0–16.1% | 13.3–20.0% |
| Upsample forward | 13.0–13.8% | 13.6–13.9% | 13.8–22.0% |
| Global forward | 38.8–39.4% | 37.3–37.7% | 33.3–33.3% |
| Irregular full backward | 13.2–13.7% | 11.4–11.4% | 12.1–12.6% |
| Upsample JVP | 17.0–17.9% | 16.8–19.9% | 18.7–18.8% |
| Complete CNN/SGD step | 5.3–6.2% | 3.0–5.8% | 7.1–7.4% |

Negative values denote regressions. Full per-process samples, same-source
composition comparisons and composition-control drift are retained in
`iteration-41.json` and its harness; individual rows must not be generalized into
a uniform speedup. Complete-training differences are small, and browser
composition-control drift can exceed them; these measurements do not establish
a stable browser training-throughput gain. Eight additional paired CPU/WASM runs cover 21 existing
library workflows, and sixteen alternating fresh CPU processes measure namespace
imports with NumPy included/preloaded separately. Source footprint estimates
exclude browser transport, package installation and total process memory.

## Evidence and release policy

- `packages/stridetorch/benchmarks/results/iteration-41.json` and `iteration-41-harness.tar.gz`.
- `packages/stridetorch/benchmarks/results/iteration-41-integration.json` and `iteration-41-integration-harness.tar.gz`.
- Reusable benchmark: `packages/stridetorch/benchmarks/bench_average_pooling_1d.py`.

Only `__init__.py`, `_api.py` and `_average_pooling.py` change among the 49
runtime/license resources; 46 remain byte-identical. Canonical tag/manual workflow
and publishing-policy bytes match their established baseline. No tag, workflow
dispatch, package upload or online deployment is requested or performed. The user
will release later. This iteration implements CPU/NumPy and Pyodide WASM;
WebGPU residency, transfer semantics and end-to-end graph execution remain future work.

Latest upstream `dbe30b4806e7d07e91dae399fb3fdad8e2e21926` adds 25 frontend/check/style/documentation
files. These changes were retained by fast-forward and validated with the full
frontend suite and both typed builds. Package dependencies, Python, SDK, catalog,
runtime and publishing-policy inputs did not change. All 49 source/license and
161 export identities retain the numerical/full-WASM proof. See
`iteration-41-main-sync.json` and `iteration-41-main-sync-harness.tar.gz`.

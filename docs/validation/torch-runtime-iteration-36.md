# StrideTorch 0.36: inverse from a Cholesky factor

Version 0.36 adds `torch.cholesky_inverse(input, upper=False)` and the matching
Tensor method. It remains an independent NumPy-only runtime, with no application,
native Torch or SciPy runtime dependency. The package name stays `stridetorch`
pending the user's browser-oriented naming choice; nothing has been published.

## Values, saved arrays and full native derivatives

The selected lower/upper factor produces the inverse of its positive-definite
system, without reconstructing the coefficient. Batches, strided inputs, negative
nonzero real diagonals, empty matrices and empty batches are covered. Outputs
have column-major final axes. A zero factor diagonal reports the first failing
batch and element. Complex64/128 support inference with real diagonals. Non-real
diagonals are invalid Cholesky factors: native POTRI's artifacts there change
with LAPACK blocking, so this library reports an explicit unsupported error.
Half, complex AD, output buffers and general higher-order reverse retain explicit
boundaries rather than silently selecting a different backend.

For a lower factor F and inverse X, the VJP is
−X (G + Gᵀ) X F; upper factors place F on the left. Like native, the primal
reads the selected triangle but differentiation extends the full factor.
Input and output alias versions guard the saved factor and inverse. Only those
arrays survive the primal call; the temporary triangular inverse is released.

The [native source](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/csrc/autograd/FunctionsManual.cpp#L2170)
and installed Torch 2.14.1 agree on a distinct ordered JVP. Define
C = Fᵀ and D = DF for lower factors, or C = F and D = DFᵀ for upper factors.
Then K = −X D X C and T = K + Kᵀ. For noncommuting factors this is different
from differentiating the mathematical inverse system. The recorded 2×2 native
probe demonstrates that difference numerically. We preserve native compatibility,
including reverse through both primal and direction channels, and do not label
the mathematical alternative as the same contract. Independent tests include
an exact noncommuting fixture and finite differences of the ordered SPD-chain
expression; ordinary inverse training uses its mathematical gradient.

One analytical three-input tangent node shares intermediate products and keeps
the full X/F/DF pullbacks. It guards DF only when another active adjoint needs
its values. A frozen factor therefore permits editing an unneeded direction
source, without copying it. The tangent itself is not saved by this linear
pullback; downstream operations retain their own guards. No factor, inverse,
solution, direction or seed is cached between calls. Empty primal strides stay
column-major, while native's empty tangent uses conventional C strides.

## Measured primal paths and retained development findings

The initial one-triangular-inverse/Gram implementation regressed small inference
by about 43–48% and large WASM inference by about 28–34%. These results and
source snapshots remain in the archive. Direct 1×1/2×2 formulas, one symmetric
copy instead of several masked allocations, and NumPy's small inverse entry
remove much of that cost. Small factors avoid constructing an identity matrix.
Larger CPU factors use one triangular inverse and a Gram product; measured
Emscripten factors above 64 rows use two blocked triangular solves. These are
performance choices with the same public value/gradient domain. Remaining
inference overhead is retained below, not described as a uniform speedup.

Nonfinite inverse values trigger structural column solves and a triangular
product that omits mathematical zero terms. This preserves finite trailing
blocks instead of spreading a NaN through a dense zero product. Native tests
cover real NaN/infinite diagonals and unused-half NaNs. These checks do not
promise identical artifacts for every invalid nonfinite complex factor across
LAPACK vendors. The first focus exposed empty tangent strides, overly broad
direction guards and upper nonfinite contamination; all failure logs are kept.
An initial standalone collection error omitted two parameter names and was
corrected before the final gates.

## Verified local scope

- Full merged Python suite: **32,463 passes, 1 optional skip,
  34 warnings, 389.36s**.
- Focused factor/inverse/solve native and standalone contracts: **1,314 passes**.
  Native comparisons validate values, strides, full VJP/JVP reverse channels,
  retained seeds, allowed edits and required mutation errors.
- Node Pyodide and actual Chrome Worker each pass **1,532 checks in 35 suites**
  on pinned Pyodide 0.29.4 / NumPy 2.2.5 without native Torch.
- Independent wheel on Python 3.10.19 / NumPy 1.26.4: **1,544 passes**.
  All 47 core/license resources agree across public/private distributions.
  Coexistence, aliases, CLI, ZIP optimizer resume, the new inverse/JVP/reverse
  API in both namespaces and independent uninstall pass.
- Original native application catalog: **78 problems, 10,218 executions,
  342 mutants, 3 repeats, 141.02s**, unchanged budgets.
- Actual exported WASM: **78/78 modules, 342 mutants,
  257119ms**, concurrency 2, no deadline exceeded.
- Current full frontend check and both typed builds pass. All **8,649** previous
  immutable assets remain byte-identical; **158** new payloads are reachable and
  all **161** active index/payload files are SHA-256 frozen.

The first full frontend run and an isolated repeat found an upstream profile
navigation check reading `opening` after its fixed 10ms delay. Diagnostic state
showed the actual route/component reaches `reading`. Two bounded waits now
require the real reading/returning route and component, retaining all assertions
and a two-second failure limit. Three targeted runs and the complete frontend
rerun pass. The original failed receipts and exact test diff are archived.
Application UI behavior is unchanged by that test synchronization repair.

## Paired whole-workflow measurements

Eight sequential children cover two dtypes and two fresh processes per CPU/WASM
after heavy gates. Each uses nine ABBA/BAAB blocks, warmups, GC enabled, actual
SGD updates and output/dtype/gradient/state checks. The baseline is the existing
0.35 public `cholesky_solve(I,F)` composition with an already-created identity.
For JVP benchmarks the baseline explicitly evaluates native inverse's ordered
expression; using solve's own JVP would compare different contracts. Both sides
project benchmark F/DF; separate native tests validate full unmasked semantics.

| Whole workflow | CPU time reduction | WASM time reduction |
| --- | ---: | ---: |
| tiny-inference | 21.8 to 23.8% | 17.6 to 19.9% |
| small-inference | -8.2 to -2.5% | 2.3 to 7.9% |
| large-inference | 26.3 to 31.0% | -4.4 to -3.9% |
| upper-inference | 27.0 to 31.8% | -4.4 to -3.6% |
| batch-inference | 38.0 to 40.2% | -1.6 to 0.4% |
| tiny-backward | 26.2 to 26.9% | 22.5 to 23.8% |
| small-backward | 14.8 to 17.6% | 19.4 to 24.3% |
| large-backward | 50.0 to 56.1% | 5.1 to 8.6% |
| sgd-factor | 21.3 to 23.5% | 20.4 to 23.3% |
| spd-chain | 28.2 to 31.3% | 11.1 to 12.4% |
| jvp | 6.5 to 9.9% | 11.2 to 12.2% |
| large-jvp | 23.3 to 30.4% | -0.1 to 0.3% |
| jvp-backward | 15.8 to 17.1% | 16.4 to 18.1% |
| large-jvp-backward | 28.2 to 32.3% | 5.2 to 6.6% |
| fixed-factor-jvp | 13.6 to 18.7% | -0.9 to -0.2% |
| generic-control | 0.3 to 2.3% | -0.7 to 0.8% |
| cholesky-control | -1.0 to 0.2% | -0.2 to 0.4% |
| unary-control | -2.4 to 0.7% | 0.3 to 1.1% |

Negative numbers mean slower. Per-call identity construction and singular
validation are included in the inverse API. Unchanged controls remain visible.
The old 0.35 APIs are retained; this new API does not replace existing solve
behavior. These shared-host measurements request thread limits but do not verify
vendor thread counts or establish statistical guarantees. They are not
native-Torch or cross-runtime throughput comparisons.

Cold CPU namespace import medians (four fresh alternating processes per set):
NumPy included 82.58 →
82.97ms; preloaded
59.87 →
65.77ms. Source bytes
517869 → 525692;
local gzip estimates 122185 →
123816. These are not browser
wire sizes. No WebGPU backend or WebGPU speedup is claimed in this version.

## Reproduction and release boundary

[`iteration-36.json`](../../packages/stridetorch/benchmarks/results/iteration-36.json)
and [`iteration-36-integration.json`](../../packages/stridetorch/benchmarks/results/iteration-36-integration.json)
record exact sources/tests, receipts, all three pilots and hashed reproduction
archives. Application gates use upstream `13d2782a494415ba72bc604a0ef7e4c4f875ba25`; the private wheel
includes its current application catalog. Latest-main synchronization and
non-force adoption are recorded separately.

The canonical [explicit-release policy](../releasing.md) stays unchanged.
No CI, tag, workflow dispatch, package upload or online publication is requested.
The user will explicitly tag later. Local success does not assert remote Linux
CI or live deployment. Broader core coverage, residual inference costs, the
0.35 quarter-width reverse gap and a resident-storage WebGPU evaluation remain
future work toward independent-library readiness.

Final synchronization preserves all five incoming UI/docs/check files through
`1f042878a36a2340d4efe1a1afe0d5ab1bdf214e`, with only the separately recorded bounded navigation
test waits differing from upstream. The complete frontend check and both typed
builds pass again. All 47 computational resources and 161 exports remain
unchanged; see `iteration-36-main-sync.json`.

Validated source was adopted non-force on remote `main` at `8666b6a90c9e374958aa71f4be0af97b8a96798c`.
A read-only query found zero Actions runs for this commit. See
`iteration-36-main-adoption.json`; no tag, dispatch or online release was created.

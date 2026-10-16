# StrideTorch 0.35: reuse Cholesky JVP solve intermediates

Version 0.35 improves the narrow real float32/64 `cholesky_solve` JVP and its
four-input reverse rule. It remains an independent NumPy-only runtime. The
current package name stays `stridetorch` pending the user's choice of a new
browser-oriented name; no rename or public package release is implied.

## Numerical rule and dispatch

Let S(R) denote two triangular solves using F's selected triangle and X = S(B).
For a lower factor, the full native derivative uses H = DF Fᵀ + F DFᵀ and
T = S(DB − H X). Upper factors transpose the corresponding products. The
[native derivative rules](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/csrc/autograd/FunctionsManual.cpp#L4719)
are validated against installed native Torch 2.14.1, including the unused half
of F and DF. This is a compatibility implementation, not native Torch in WASM.

When F is actually finite triangular, the first primal solve already provides
Y = FᵀX (lower) or FX (upper). For the lower case, compute
U = solve(F, DB − DF Y), then T = solve(Fᵀ, U − DFᵀX).
The upper case computes U = solve(Fᵀ, DB − DFᵀY), then
T = solve(F, U − DFX). This replaces two factor-times-solution products with
already available arrays. The adjoint's first solve similarly supplies FᵀQ/FQ,
and the second tangent RHS supplies FᵀT/FT.

These identities optimize values at the triangular factor. The analytical
pullback still has all four inputs X, F, DB and DF and the full unmasked factor
and direction edges. Differentiating only the simplified triangular expression
would lose those edges. No numerical values or seeds are cached across calls.
Conditional direction snapshots, alias mutation guards and selective saved-array
ownership from 0.34 remain in force.

The narrow branch requires n ≥ 64 and RHS width ≤ n/4. General full factors
retain 0.34's cached-product rule. Nonfinite intermediate checks fall back to
the prior product/residual/solve ordering; small/wide cases keep their existing
rule. No API, dtype or gradient domain is removed to obtain speedups.
Complex inference remains supported; complex AD, half factorization, output
buffers and general higher-order reverse retain their documented boundaries.

The first prototype regressed CPU frozen-factor and small JVP flows by about
4–6%. Public validation now dispatches directly to a private validated primal
helper, avoiding recursive validation. Gradient factors already require the
triangle scan for their VJP; new scans for frozen factors are enabled only on
the measured Emscripten runtime, and not for n=64 with RHS width ≤4. CPU frozen
factors retain the prior unclassified rule. WASI and WebGPU receive no new
performance claim. All four pilot source/result sets are retained.

## Local correctness and application validation

- Complete merged Python suite: **32,235 passes, 1 optional skip,
  34 warnings, 325.37s**.
- Focused Cholesky, native-reference, mutation, ownership and retained-reverse
  tests: **859 passes**. New cases cover both dispatch boundaries, ignored-half
  nonzero entries, full/triangular directions, batches, strided final axes,
  both triangles and extreme positive/negative scales in float32/64.
- Independent exact scaled adjoints and constructed no-reverse/no-grad systems
  complement native comparisons. Node Pyodide and actual Chrome Worker each
  pass **1,495 checks in 34 suites** on pinned Pyodide 0.29.4 / NumPy 2.2.5.
- Installed independent wheel: **1,507 passes** on Python 3.10.19 / NumPy 1.26.4
  without native Torch. Public/private wheel ownership, all 47 core/license
  resources, aliases, CLI, ZIP training resume and independent uninstall pass.
- Original native catalog: **78 problems, 10,218 executions, 342 mutants,
  3 repeats, 185.03s**, unchanged budgets.
- Actual exported WASM: **78/78 modules, 342 mutants,
  238969ms**, concurrency 2, no deadline exceeded.
- Full frontend check and both typed builds pass. All **8,491** previous
  immutable assets remain byte-identical; **158** new payloads are reachable,
  and all **161** active index/payload files are SHA-256 frozen.

## Paired whole-workflow measurements

Sixteen sequential children cover two dtypes and two fresh processes per
runtime and comparison, after the heavy correctness/native/frontend gates.
Each uses nine ABBA/BAAB paired blocks, warmups, GC enabled, real SGD updates,
and value/dtype/gradient/state checks. The 20 workflows include fixed-factor and
no-reverse cases that exposed the initial regression. Native tests separately
cover full unmasked derivative semantics outside the projected benchmark domain.

| Workflow | CPU time reduction vs 0.34 | WASM time reduction vs 0.34 |
| --- | ---: | ---: |
| jvp | -10.6 to 1.4% | -0.3 to 0.2% |
| narrow-jvp | 0.7 to 3.0% | 10.1 to 12.0% |
| narrow-jvp-backward | -0.6 to 4.8% | 1.8 to 3.3% |
| quarter-jvp | 0.7 to 3.1% | 24.2 to 25.2% |
| quarter-jvp-backward | 1.2 to 2.9% | 16.8 to 17.7% |
| batch-jvp-backward | -1.4 to 8.7% | 5.0 to 6.7% |
| fixed-factor-jvp | -0.6 to 1.1% | 3.5 to 3.9% |
| fixed-factor-jvp-backward | -0.2 to 1.3% | 16.9 to 17.4% |
| no-reverse-jvp | -0.1 to 1.1% | 22.2 to 25.2% |
| no-reverse-tiny-jvp | 0.3 to 2.4% | 0.1 to 3.9% |
| generic-control | -0.8 to 3.1% | -0.5 to 1.5% |
| cholesky-control | -4.8 to 1.3% | -1.3 to 0.3% |
| unary-control | -2.3 to 1.1% | -0.3 to 1.1% |

Negative values mean slower. Against the separately retained 0.32 public
two-triangular-solve composition, WASM quarter-width forward JVP changes by
**2.6 to 4.0%** and its full reverse by
**-7.5 to -6.2%**. These older-baseline
comparisons remain visible so a gain against 0.34 does not hide inherited costs.
The older-composition frozen-factor full reverse changes by
**-7.9 to -5.0%** on WASM;
these remaining reverse-workflow gaps are still optimization targets.

The initial CPU set includes a -10.6% small-JVP result and sizable unchanged
control variability, with some absolute latencies moving by nearly a factor of
two between children. Four additional complete paired CPU children run after
the exported WASM gate, with unchanged source and all original data retained.
Their small JVP range is **0.4 to 1.9%**, tiny backward
**-1.2 to 0.2%**, and unchanged Cholesky control
**-0.6 to 0.2%**. This follow-up assesses the unresolved
small-workflow concern; it does not replace the original table or establish a
uniform CPU gain.

Cold CPU namespace import medians (four fresh alternating processes per set):
NumPy included 87.13 →
90.43ms; preloaded
59.86 →
59.93ms. Source bytes
515467 → 517869;
local gzip estimates 121531 →
122185. These shared-host
measurements request thread limits but do not verify vendor thread counts.
They are not statistical guarantees, browser wire sizes or CPU/WASM throughput
comparisons. No WebGPU backend is enabled by this iteration.

## Reproduction and adoption boundary

[`iteration-35.json`](../../packages/stridetorch/benchmarks/results/iteration-35.json)
and [`iteration-35-integration.json`](../../packages/stridetorch/benchmarks/results/iteration-35-integration.json)
record exact source/test hashes, receipts, initial pilots and hashed reproduction
archives. Application checks use upstream `e21b136c2431d2aeeea52b0d1b1124a09c413f6a` including its
current catalog in the rebuilt private wheel.

Local gates do not establish remote Linux CI or online deployment. The
canonical [explicit-release policy](../releasing.md) remains unchanged. No tag,
workflow dispatch, package publication or CI run is requested by this iteration;
the user will explicitly release later. Main adoption is recorded separately.

Final main synchronization preserves all six upstream UI/docs/check files through
`3a5673cae508a41814c8dbfea63787e3aaaa0e83` byte for byte. Twenty affected mounted UI checks and both
typed builds pass. The 47 computational resources, 161 exports, Python/catalog,
lock and WASM driver inputs remain unchanged. See `iteration-35-main-sync.json`.
An initial harness preparation used an unresolved relative template path; no UI
test ran until the path was corrected. Both terminal logs are retained.

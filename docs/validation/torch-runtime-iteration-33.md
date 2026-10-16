# StrideTorch 0.33: Cholesky solve and portable failure reporting

StrideTorch adds `torch.cholesky_solve(input, input2, upper=False, *, out=None)`
and `Tensor.cholesky_solve(input2, upper=False)`. It remains an independent
NumPy-only runtime with the `stridetorch` launcher and explicit `torch` aliases;
PaperLeet consumes the same source through a private installed namespace and
browser bundles. No native Torch or SciPy is needed at runtime.

The implementation follows the [official API](https://docs.pytorch.org/docs/2.14/generated/torch.cholesky_solve.html)
and [native differentiation rules](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/csrc/autograd/FunctionsManual.cpp#L4719).
The actual keyword names `input`/`input2` were verified against installed native
Torch 2.14.1. Factor/RHS batches broadcast and final matrix axes use native
column-major layout. Float32/64 support analytical VJP, JVP and reverse through
both JVP channels; complex64/128 support inference. Half, complex AD and `out`
remain explicit unsupported cases.

Primal values consume only the selected factor triangle. Native differentiation
extends the full factor and direction, including the unused half. Full unmasked
gradients are retained. RHS, factor and output mutation guards remain active,
including when just one input needs gradients. Unnecessary RHS arrays are not
retained.

## Numerical portability

The previous 0.32 remote native and exported WASM jobs passed, but its
infrastructure job failed four NaN exception assertions; Pages build/deploy
were skipped. Exact evidence is retained in `iteration-32-remote-failure.json`.
The last positively verified live deployment remains 0.31.

Native nonfinite linear-algebra behavior is [not guaranteed](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html#non-finite-values).
Finite invalid/singular cases keep strict native comparisons. NaN rejection is
an explicit adapter boundary rather than a universal native exception promise.
Real Pyodide/NumPy 2.2.5 probes exposed quiet NaN factors for finite non-positive-
definite inputs and quiet singular solve/inverse failures, including empty RHS.

Cholesky now requires positive returned real factor diagonals. Exceptional
batch/leading-minor diagnostics use the same check without reading ignored input
entries. Positive infinite diagonals retain their tested behavior. A restored-
error-mode import probe records only whether NumPy traps invalid floating-point
status. On quiet runtimes, suspicious NaN general solve/inverse results use
`slogdet` zero-sign status; empty RHS solves always check singular status. Normal
CPU solves avoid the extra scan/factorization. No tensor solutions are cached.
The triangular API permits nonfinite zero-pivot results, so tests do not assert
one vendor-specific NaN/Inf pattern.

## Complete local verification

- Full merged application and native-reference core suite: **31,655 passed,
  1 optional Docker skip, 34 warnings, 390.81s**.
- Focused new factor/solve/portability tests: **776 passed**.
- Node Pyodide and actual Chrome 155 Worker: **1,375 checks in 34 suites each**;
  Pyodide 0.29.4 / NumPy 2.2.5, no native Torch.
- Installed independent wheel, Python 3.10.19 / NumPy 1.26.4: **1,387 passed**,
  no native Torch. Public/private wheels have identical 47 core/license resource
  hashes, disjoint installed paths, and successful co-install/uninstall,
  launcher/alias, ZIP checkpoint and optimizer-resume checks.
- Original native PaperLeet catalog: **78 problems, 10,218 test executions,
  342 mutants, 3 repeats, 119.67s** under unchanged budgets.
- Actual exported WASM catalog: **78 modules, 342 mutants, 247505ms**,
  concurrency 2, all original browser deadlines retained and satisfied.
- Complete frontend check and both development/Pages-profile builds pass,
  including 63 bilingual guides / 189 prompts and canonical release-policy tests.
- All **8,175** prior immutable main assets are byte-identical; all **158** new
  payloads are reachable. All **161** active index/payload files are frozen by
  SHA-256, and both runtime bundle profiles carry LICENSE/NOTICE.

Initial Node/Chrome runs exposed 12 invalid-factor failures; those raw sources
and logs remain archived. The first final private application wheel had a stale
merged catalog and failed identity verification; the artifact was retained and
rebuilt. A native run concurrent with infrastructure/frontend passed numerical
and mutation checks but exceeded one flow-matching per-test budget at 11.54s;
the complete catalog was rerun in isolation without changing any budgets.
These failures are part of the evidence, not successful gate receipts.

## Measured efficiency

The VJP shares one adjoint solve. For actual finite triangular factors with at
least 64 rows, it reuses one graph-local forward intermediate and one adjoint
intermediate in two matrix products. General finite narrow factors associate
through RHS width; finite narrow JVPs avoid forming a dense coefficient
direction. Nonfinite/general fallback paths retain the original expression.
Saved graph intermediates are released with the graph; there is no cross-call
factor/solution cache. The large narrow-JVP workload remains slower than the
0.32 composition (about 16–19% on WASM and 1–4% on CPU); it remains an explicit
optimization target rather than a uniform speedup claim.

The release comparison uses the same projected factor, with 0.32's two public
triangular solves versus 0.33's fused API. This is a whole-workflow comparison,
not a speedup over a missing 0.32 API. Sixteen sequential timing children cover
two dtypes and two fresh processes per CPU/WASM/lane after heavy tests complete.
Each uses nine paired ABBA/BAAB blocks, GC enabled, warmups, real SGD updates and
numeric/dtype/gradient/state assertions. Values below are ranges of median time
reductions; negative values mean slower. Shared-host noise and effective BLAS
thread counts remain limitations; CPU and WASM throughput are not compared.

| Workflow | CPU time reduction | WASM time reduction |
| --- | ---: | ---: |
| tiny-backward | 18.7 to 20.9% | 20.2 to 21.6% |
| sgd-factor | 10.3 to 15.6% | 15.6 to 17.7% |
| batch-b-only | 4.2 to 4.6% | 5.1 to 8.0% |
| narrow-backward | 8.4 to 10.3% | 6.4 to 7.3% |
| upper-narrow-backward | 8.9 to 9.6% | 6.8 to 8.9% |
| wide-backward | 10.3 to 12.4% | 5.2 to 28.1% |
| spd-chain | 3.2 to 4.4% | 0.1 to 1.0% |
| jvp | 16.1 to 18.4% | 21.6 to 23.7% |
| narrow-jvp | -3.4 to -1.5% | -18.7 to -15.5% |

A separate formula lane compares the initial 0.33 full-factor rule with the final
association: WASM narrow factor backward changes by
87.3 to 88.2%, upper narrow by
87.6 to 88.2%, and narrow JVP by
50.6 to 55.0%. All initial raw measurements are retained:
they exposed narrow-JVP and some backward regressions that prompted the final
refinement. Unchanged generic solve, Cholesky and unary controls, cold imports,
source bytes and local gzip estimates are recorded in the JSON report. Gzip
estimates are not measured browser transfer sizes.

## Adoption and release

Reports are [`iteration-33.json`](../../packages/stridetorch/benchmarks/results/iteration-33.json)
and [`iteration-33-integration.json`](../../packages/stridetorch/benchmarks/results/iteration-33-integration.json),
with hashed reproducibility archives. Main adoption is tracked separately from
these local gates. No CI, tag, workflow dispatch or online publication has been
triggered by this iteration. Canonical upstream [explicit release policy](../releasing.md)
is preserved: ordinary pushes/PRs are quiet; a user-created stable current-main
version tag or explicit manual release runs all quality checks before Pages.
The user plans to tag later. Local success does not assert remote Linux CI or a
new live deployment. WebGPU remains a separate end-to-end evaluation; this
version uses CPU/WASM.

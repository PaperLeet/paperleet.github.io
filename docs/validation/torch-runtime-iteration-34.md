# StrideTorch 0.34: fused narrow JVP and native direction snapshots

Version 0.34 improves real float32/64 `cholesky_solve` JVPs with at least 64
factor rows and RHS width at most one quarter of the factor size. It keeps the
full native factor/direction extension and reverse differentiation through both
JVP channels. This is an independent NumPy-only runtime; PyTorch remains a test
oracle. The package name remains `stridetorch` while a browser-oriented replacement
name is being considered; no new name has been adopted or published.

## Fused analytical rule

Let S(R) denote two triangular solves using the selected triangle of F.
For a lower factor, X = S(B), H = DF Fᵀ + F DFᵀ and T = S(DB − H X).
The products in H use the full F/DF, including the unused half, as required by
native differentiation. Upper factors transpose the corresponding products. This follows the
[native rules](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/csrc/autograd/FunctionsManual.cpp#L4719)
checked against installed native Torch 2.14.1.

The new tangent node has four inputs: X, F, DB and DF. For tangent seed G,
Q = S(G). Its pullbacks are −H Q for X, Q for DB,
−(Q Xᵀ + X Qᵀ) F for DF, and the sum of the outer-factor solve adjoint and
−(Q Xᵀ + X Qᵀ) DF for F. The primal X graph carries the remaining B/F
contributions. Products associate through RHS width, avoiding dense cubic
coefficient-direction products and the chain of intermediate Tensor nodes.
Forward F/DF-times-X products and adjoint F/DF-times-Q products are shared
across the four pullbacks. The first fusion recomputed them per input and
regressed 20–25% on quarter-width tangent backward; the final rule removes
those duplicate products without narrowing the supported input domain.
All broadcast reductions, full unmasked factor gradients and zero reverse edges
remain covered. Nonfinite F/DF/X, factors with fewer than 64 rows and wider RHSs keep the
existing JVP implementation;
complex AD, `out`, half and general higher-order reverse remain unsupported.

## Mutation and saved-array ownership

180 native probes cover all 15 nonempty gradient-activity combinations, both
triangles and edits to B, F, DB, DF, X and T. The 0.33 baseline rejects 12 cases
that native permits. The first fusion still rejects eight; a conditional DF
snapshot fixes the distinction. When F needs no adjoint but X does, native saves
the coefficient direction and permits editing its DF source. We snapshot that
source without forming H. When the factor adjoint needs DF, its version guard
remains active. RHS directions enter linearly and need no value guard.

Unneeded X and T arrays are not retained by the new pullback. Version counters
still detect required edits through aliases. Active gradient leaves retain the
engine's existing ownership; they are not falsely required to disappear while
the graph needs them. Weak-reference tests check release before/after backward,
permitted direction edits and actual original-value gradients. There is no
cross-call tensor, factorization, solution or seed cache.

## Verified local scope

- Full infrastructure/native-reference suite: **32,043 passes,
  1 optional Docker skip, 34 warnings,
  256.31s**.
- Focused solve/JVP/native mutation/broadcast/empty/strided/no-grad/retained
  reverse/ownership tests: **667 passes**.
- Node Pyodide and actual Chrome Worker: **1,439 checks in 34 suites each**, on
  Pyodide 0.29.4 / NumPy 2.2.5, without native Torch.
- Installed independent wheel on Python 3.10.19 / NumPy 1.26.4: **1,451 passes**,
  no native Torch. Public/private wheel core/license hashes agree for all 47
  resources; install/uninstall, aliases, CLI, ZIP resume and large fused JVPs in
  both isolated namespaces pass.
- Original native catalog: **78 problems, 10,218 executions, 342 mutants,
  3 repeats, 123.44s**, run alone with unchanged budgets.
- Actual exported WASM: **78/78 modules, 342 mutants, 321345ms**,
  concurrency 2, no browser deadline exceeded.
- Current frontend check and both builds pass; all **8,333** prior immutable
  assets remain byte-identical, all **158** new payloads are reachable, and all
  **161** active index/payload files are frozen by SHA-256.

The first expanded focus had 627 passes and four ownership-test failures. The
test had consumed its graph before a second backward and incorrectly expected
active leaves to disappear. Retained-graph/leaf expectations were corrected,
and the implementation was improved to release genuinely unneeded X/T arrays.
All raw probes, first timing results, source snapshots and failures are retained.
The first full set of 16 timing children exposed a wider tangent-backward
regression. The remaining old-source WASM run was deliberately terminated; it
is preserved as partial evidence, not a successful final gate.

## Paired whole-workflow timings

Sixteen sequential children cover two dtypes and two fresh processes per CPU/WASM
and comparison. Each uses nine ABBA/BAAB paired blocks, warmups, GC enabled,
real SGD updates and value/dtype/gradient/state checks. The primary lane compares
0.33's fused API with 0.34 through the same projected factors and directions.
The explicit JVP-backward workload also differentiates both output channels
through all four original inputs. Native tests separately validate full unmasked
semantics beyond that projected benchmark domain.

| Workflow | CPU time reduction vs 0.33 | WASM time reduction vs 0.33 |
| --- | ---: | ---: |
| narrow-jvp | 14.4 to 17.3% | 14.8 to 20.1% |
| narrow-jvp-backward | 11.4 to 14.7% | 12.3 to 13.7% |
| quarter-jvp | 13.5 to 15.2% | 5.0 to 8.6% |
| quarter-jvp-backward | 7.4 to 13.3% | -1.4 to 0.4% |
| batch-jvp-backward | 9.2 to 12.0% | 9.0 to 11.6% |
| generic-control | 0.1 to 1.7% | -2.8 to 1.2% |
| cholesky-control | -1.5 to -0.0% | -0.4 to 1.2% |
| unary-control | -1.6 to 2.4% | -1.3 to 1.1% |

Negative values mean slower. The old 0.32 two-triangular-solve composition is a
separate lane: narrow JVP changes by 12.5 to 14.6%
on CPU and -1.4 to 10.4% on WASM. This retains the
comparison that exposed the 0.33 regression. At quarter RHS width, the gap to
that older composition remains -31.0 to -18.0% for
WASM forward JVP and -28.4 to -9.9% for
its full reverse. This inherited gap remains an optimization target. The final
quarter-width reverse range versus 0.33 includes zero and is treated as roughly
unchanged, not a demonstrated speedup. One old-composition unary control sample
also shows a larger outlier; unchanged controls and all raw blocks are retained.
All other backward/SGD/small-JVP
workflows and unchanged controls are in the JSON report; a new JVP optimization
does not imply gains for unchanged ordinary backward code.

Cold namespace import medians, four fresh alternating processes each:
NumPy included 139.85 →
141.82ms; NumPy preloaded
102.80 →
98.28ms. Source bytes change from
512093 to 515467; local
gzip estimates from 120643 to
121531. These are shared-host
measurements with requested thread limits, not verified BLAS thread counts,
statistical guarantees, browser wire sizes or native-Torch/CPU-WASM comparisons.

## Integration and release

[`iteration-34.json`](../../packages/stridetorch/benchmarks/results/iteration-34.json)
and [`iteration-34-integration.json`](../../packages/stridetorch/benchmarks/results/iteration-34-integration.json)
include source/test hashes, exact receipts and hashed reproduction archives.
The full frontend check and both builds pass at `27d9e600`. Ten later UI-only
files from `c8d5baac` are preserved byte for byte, with all 47 computational
resources and 161 exports unchanged. Fourteen affected UI checks and both typed
builds pass after that merge; source review does not claim manual visual QA.
See `iteration-34-main-sync.json`. Its first runner had the wrong working
directory; only that runner was corrected and the failure log retained.
Latest-main synchronization and non-force adoption are recorded separately.
No CI, version tag, workflow dispatch or online release is triggered by this
iteration. The canonical [explicit-release policy](../releasing.md) is retained;
the user will tag later. Local success does not assert remote Linux CI or a new
live deployment. WebGPU remains an end-to-end evaluation target, not an enabled
backend in this version.

The final main sync also preserves all seven incoming reading-pane files from `dd6ca81f`. Its new reply-viewport check, the 14 related checks and both typed builds pass. All computational resources, exports and Python/native/WASM inputs remain unchanged; see `iteration-34-main-sync-final.json`.

The validated source was adopted non-force on remote `main` at `ded40931`.
A read-only post-push query found no Actions run for that commit. See
`iteration-34-main-adoption.json` for the exact SHA and verification time.
No tag, workflow dispatch or online release was created; subsequent proof-only
commits do not change the validated runtime or exports.

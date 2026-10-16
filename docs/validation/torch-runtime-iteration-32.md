# StrideTorch 0.32 Cholesky factors

This release adds `torch.linalg.cholesky(A, *, upper=False)` on the existing
NumPy CPU/Pyodide backend. Complete local integration is verified below;
remote Quality, actual deployment and live checks remain separate. The 0.31
release retains its own pinned source/export identities and verified live state.

The API follows the [official Cholesky contract](https://docs.pytorch.org/docs/2.14/generated/torch.linalg.cholesky.html).
Native differentiation is checked against [PyTorch's primary implementation](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/csrc/autograd/FunctionsManual.cpp#L2125).
The implementation uses NumPy factorization and StrideTorch's triangular solves;
it requires neither native Torch nor SciPy at runtime.

Real float32/64 support batched values, reverse gradients, JVP and reverse
differentiation through both the primal and direction. Complex64/128 support
inference only. Half/integer factorization, complex differentiation and `out`
buffers remain explicit unsupported cases.

Only the selected input triangle is consumed, including when the unused half
contains NaN. Both choices return column-major final axes, including empty and
singleton shapes. The reverse tape saves the output factor, not A: modifying A
after forward is allowed; modifying the factor invalidates backward. Failed
factorizations resolve the first failing batch and leading minor on the error
path. Native JVP uses an untransposed direction for either `upper` choice; its
outer factor remains differentiable.

The initial focused run passed133 and failed30 cases. Twenty failures came from
a test attempting to construct differentiable complex Tensors even under
`no_grad`, which the existing library intentionally rejects. Eight exposed empty
tangent strides; two exposed dense LU's nonfinite contamination during backward.
The latter two implementation issues were corrected by empty-layout
normalization and triangular substitution for infinite factor diagonals. The
corrected focused run passes163 cases, with the original failed log retained.

Independent tests use scalar two-by-two value/VJP/JVP formulas, finite
differences through an SPD parameterization, real SGD updates and checkpoint
resume. Native comparisons cover strided and batched matrices, larger systems,
non-symmetric directions, mutation guards, empty shapes and selected-triangle
nonfinite behavior. Complete core/package/WASM gates and measured refinements are recorded below; application integration remains a separate gate.


## Validated core and package

The refined source passes31042 infrastructure/library tests with one optional
Docker skip and34 warnings in196.96s. Focused native/independent tests pass163.
An isolated Python3.10.19/NumPy1.26.4 wheel environment without native Torch
passes1140. Node and actual Chrome155 Worker each pass1128 independent checks
across32 suites under Pyodide0.29.4/NumPy2.2.5. All47 Python/license resources
match frozen source hashes.

The first Node/Chrome runs passed1128 with pytest exit0; their harness expected
1130 because the new test count was added incorrectly. Corrected1113+15=1128,
with all32 suites retained. First logs/plans and both corrected/refined proofs
are retained. Public/private wheels coinstall with disjoint ownership, execute
alias imports, ZIP training resume and Cholesky VJP/JVP/full tangent reverse,
and each survives uninstalling the other distribution.

## Measured refinement

The baseline is this release's first Cholesky implementation with generic NumPy
LU systems in its pullback. Version0.31 has no Cholesky API and is not used as a
throughput baseline. Both compared namespaces use the same0.32 API. Initial
structured timing improves selected large cases but regresses small backward/
SGD flows. The refinement adds analytical1x1/2x2 pullbacks and sends already
triangular factors up to64 rows directly to a single LAPACK call per system;
larger factors retain blocked triangular substitution. No tensor values or
factorizations are cached. All original timings are retained.

Final percentages below mean lower elapsed time relative to that original
generic-LU0.32 implementation. Each range spans float32/64 and two fresh
processes per dtype, with nine paired ABBA/BAAB blocks, GC enabled, warmup,
and final numerical/gradient/updated-state checks. SGD includes actual updates
and SPD construction; Gaussian loss includes whitening and log diagonals.

| Complete flow | CPU time reduction | WASM time reduction |
| --- | ---: | ---: |
|2x2 backward|21.3 to24.3%|14.3 to21.5%|
|16-row SPD SGD|-1.2 to-0.2%|-0.9 to0.8%|
|Batch4,64-row backward|-0.9 to0.4%|-0.7 to0.6%|
|256-row lower backward|20.5 to31.7%|18.6 to22.3%|
|256-row upper backward|18.8 to35.5%|0.1 to7.6%|
|64-row Gaussian backward|-0.7 to2.2%|-1.6 to0.3%|
|Unchanged JVP control|-0.1 to1.6%|1.4 to2.4%|
|Unchanged unary control|-0.8 to2.4%|-1.6 to1.0%|

Upper float64 WASM has little measurable benefit. No universal speedup,
cross-runtime throughput comparison or host/BLAS-thread causal claim is made.
All8 refined timing children and all8 initial timing children terminate0.

Compared with0.31, source/resources grow497125 to503572 bytes; local source
gzip estimate grows116891 to118682 bytes, an increase of1791. This is not a
network transfer measurement. Sixteen fresh alternating CPU processes give
namespace-import medians85.97 to86.93ms including NumPy and59.89 to59.84ms
with NumPy already imported. First-import outliers are retained; zipfile remains
unloaded.

[Core evidence](../../packages/stridetorch/benchmarks/results/iteration-32.json)
and its594-file archive retain raw probes, failed/corrected logs, frozen
source/tests, exact package hashes, all process receipts and all timing blocks.
The autograd browser JSON contains41 package Python modules plus4 compatibility
modules,521360 raw bytes and121635 local gzip estimate bytes. The schedules
profile retains its small source closure and explicit unsupported-AD behavior.


## Complete application integration

The original78-case catalog passes native CI with10218 tests,342 mutants,three
repeats and the original600s limit in146.10s. The actual exported runtime passes
all78 cases/342 mutants under pinned Pyodide in243138ms at concurrency2,
including1128 independent contracts across32 suites. No browser calibration,
learner or total budget is exceeded; no original test,mutant or budget is removed.

Full frontend checks pass60 maintained bilingual guides/180 prompts and the
existing navigation/offline/editor/workspace contracts. Default and actual
hosted Pages-profile builds each have three initial JS assets:404162 raw/130579
gzip and405564 raw/131014 gzip bytes,within original budgets. No TSX,backend,
judge/problem,dependency or lockfile edits accompany this release.

All7862 previously published immutable assets from main b3b56a19 remain
byte-identical. All158 new payloads are reachable;161 active payload/index
files are frozen. Both profiles carry version0.32.0 and redistribution notices.
Public/private distribution hashes agree on all47 resources;the application
wheel/sdist also match current server/catalog bytes and preserve disjoint wheel
ownership.

[Application integration evidence](../../packages/stridetorch/benchmarks/results/iteration-32-integration.json)
and its206-file archive retain actual full results,all gate receipts,exports,
dependency identities and source hashes. Core source is5c5ed0f4;non-force main
adoption revalidates ancestry and exact remote ref. The last positively verified
live site remains0.31 atb3b56a19 until0.32's exact-source remote gates pass.
